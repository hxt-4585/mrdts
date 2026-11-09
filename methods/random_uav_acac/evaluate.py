"""Paired held-out evaluation of ACAC, PPO checkpoints, own and random actions.

python methods/random_uav_acac/evaluate.py --checkpoints ACAC_RUN/checkpoint.pt \
    --ppo-checkpoints PPO_RUN/checkpoint.pt --episodes 10

Every policy is evaluated on the first ACAC checkpoint's environment snapshot,
seed-derived flight and resampling interval. Actual DAG and flight hashes are
written per episode. The PPO reader loads actor tensors without importing any
code from another baseline. Evaluation does not update model parameters.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from methods.random_uav_acac.rollout import Adapter, collect_slot
from methods.random_uav_acac.run import Rows, load_checkpoint, output_directory
from methods.random_uav_acac.scene import Summary, build_scene, settings_from_snapshot, write_json


def digest_requests(digest, requests):
    for request in requests:
        dag = request.dag
        record = [request.key, dag.node_num, dag.edges, dag.node_features.tolist(), sorted(dag.edge_features.items())]
        digest.update(json.dumps(record, separators=(',', ':')).encode('utf-8') + b'\n')


class PPOCheckpoint:
    """Inference-only reader for the reference checkpoint's public tensor layout."""
    def __init__(self, checkpoint, observation_dim, count, device):
        hidden = checkpoint['config']['hidden']
        self.device = device
        self.actors = nn.ModuleList([
            nn.Sequential(nn.Linear(observation_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                          nn.Tanh(), nn.Linear(hidden, count + 2)) for _ in range(count)
        ]).to(device)
        weights = checkpoint['member_policies']
        for owner, actor in enumerate(self.actors):
            prefix = f'{owner}.actor.'
            actor.load_state_dict({key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)})
        self.actors.eval()

    def select(self, key, observation, mask):
        logits = self.actors[key.owner_member_id](torch.as_tensor(observation, device=self.device))
        return int(logits.masked_fill(~torch.as_tensor(mask, device=self.device), -torch.inf).argmax())


def compare(checkpoints, *, ppo_checkpoints=(), episodes=10, epoch_start=1000000, seed=None,
            device='cpu', torch_threads=1, metrics_interval=10, resample_slots=None, output_root='results'):
    torch.set_num_threads(torch_threads)
    source = torch.load(checkpoints[0], map_location='cpu', weights_only=True)
    config = source['config']
    seed = config['seed'] if seed is None else seed
    interval = config['resample_slots'] if resample_slots is None else resample_slots
    settings = settings_from_snapshot(source['environment'])
    output = output_directory(output_root, 'eval', seed)
    specifications = [('own', 'own', None), ('random', 'random', None)]
    specifications.extend((f'acac_{i}', 'acac', Path(path)) for i, path in enumerate(checkpoints))
    specifications.extend((f'ppo_{i}', 'ppo', Path(path)) for i, path in enumerate(ppo_checkpoints))
    manifest = dict(environment=source['environment'], episodes=episodes, epoch_start=epoch_start,
                    seed=seed, resample_slots=interval, ordering='ERS', checkpoints={},
                    policy='deterministic argmax for learned policies; uniform legal random and own Member controls',
                    comparison='first ACAC snapshot defines all evaluation conditions; episode/slot streams are paired')
    summaries, started = {}, perf_counter()
    with (output / 'episodes.csv').open('w', newline='', encoding='utf-8') as ef, \
            (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as mf, \
            (output / 'members.csv').open('w', newline='', encoding='utf-8') as af:
        episode_rows, metric_rows, member_rows = Rows(ef), Rows(mf), Rows(af)
        for label, kind, path in specifications:
            env = build_scene(settings, episode=epoch_start)
            adapter = Adapter(env, seed, interval)
            model = selector = None
            if kind == 'acac':
                _, model, checkpoint = load_checkpoint(path, device)
            elif kind == 'ppo':
                checkpoint = torch.load(path, map_location=device, weights_only=True)
                reader = PPOCheckpoint(checkpoint, adapter.obs_dim, adapter.count, device)
                selector = reader.select
            if path is not None:
                manifest['checkpoints'][label] = dict(path=str(path.resolve()),
                    environment=checkpoint['environment'], training=checkpoint['config'],
                    training_environment_matches_evaluation=checkpoint['environment'] == source['environment'])
            total = Summary()
            for episode in range(epoch_start, epoch_start + episodes):
                env.reset(epoch_index=episode)
                generator = np.random.default_rng(np.random.SeedSequence([seed, episode, 102]))
                if kind == 'own':
                    selector = lambda key, obs, mask: key.owner_member_id
                elif kind == 'random':
                    selector = lambda key, obs, mask: int(generator.choice(np.flatnonzero(mask)))
                dag_digest, flight_digest = hashlib.sha256(), hashlib.sha256()
                summary = Summary()
                action_counts = np.zeros((adapter.count, adapter.count + 2), dtype=np.int64)
                reward_sums = np.zeros(adapter.count)
                invalid = np.zeros(adapter.count, dtype=np.int64)
                for slot in range(env.env_config.steps_per_epoch):
                    trajectory, row = collect_slot(adapter, model, deterministic=True, selector=selector,
                        request_observer=lambda requests: digest_requests(dag_digest, requests))
                    flight_digest.update(env.members.positions[:adapter.count, :2].tobytes())
                    for event in trajectory.events:
                        for owner, mask, action in zip(event.owners, event.masks, event.actions):
                            action_counts[owner, action] += 1
                            invalid[owner] += not mask[action]
                    for owner in range(adapter.count):
                        reward_sums[owner] += row[f'member_{owner}_reward_sum']
                    summary.add(row)
                    total.add(row)
                    if (slot + 1) % metrics_interval == 0 or slot + 1 == env.env_config.steps_per_epoch:
                        metric_rows.add(dict(policy=label, **row))
                episode_rows.add(dict(policy=label, episode=episode, dag_stream_sha256=dag_digest.hexdigest(),
                    flight_stream_sha256=flight_digest.hexdigest(), **summary.result()))
                for owner, counts in enumerate(action_counts):
                    decisions = int(counts.sum())
                    member_rows.add(dict(policy=label, episode=episode, member=owner, samples=decisions,
                        reward_sum=reward_sums[owner], reward_mean=reward_sums[owner] / decisions if decisions else None,
                        invalid_actions=int(invalid[owner]),
                        **{f'action_{i}_fraction': int(n) / decisions if decisions else None
                           for i, n in enumerate(counts)}))
                for stream in (ef, mf, af):
                    stream.flush()
                print(f'{label} | held-out episode {episode} | delay {summary.result()["mean_delay_s"]:.4f}s', flush=True)
            summaries[label] = total.result()
    manifest['flight_seed'] = adapter.flight.seed
    write_json(output / 'config.json', manifest)
    write_json(output / 'trajectory.json', adapter.flight.snapshot())
    write_json(output / 'summary.json', dict(policies=summaries, elapsed_s=perf_counter() - started))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--ppo-checkpoints', type=Path, nargs='*', default=[])
    for name, default in (('episodes', 10), ('epoch-start', 1000000), ('torch-threads', 1), ('metrics-interval', 10)):
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resample-slots', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-root', default='results')
    print(f'Completed: {compare(**vars(parser.parse_args()))}', flush=True)


if __name__ == '__main__':
    main()
