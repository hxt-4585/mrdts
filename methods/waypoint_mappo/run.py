"""Train static, decentralized Master waypoints with a frozen Member scheduler.

Run with python -m methods.waypoint_mappo.run. One joint deployment is one
terminal transition. Each PPO batch shares a DAG stream across deployments.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import tomllib

import numpy as np
import torch

from methods.centralized_cem.run import evaluate_job, initialize_worker
from methods.random_uav_ppo.support import settings_from_snapshot
from methods.hmappo.ppo import Policy, update
from methods.hmappo.rollout import Adapter, decide
from methods.hmappo.support import build_scene, write_json


ROOT = Path(__file__).resolve().parents[2]


def deployment(policy, inputs, deterministic=False):
    samples = decide(policy, inputs, deterministic)
    # Initial positions of owned UAVs are visible to their respective Master.
    # Zero residual gives hover. The smooth transform keeps targets in bounds.
    targets = np.zeros((len(samples[0].action) // 2, 2), dtype=np.float32)
    for sample in samples:
        owned = sample.mask[::2]
        positions = sample.observation[:targets.size].reshape(-1, 2)[owned]
        actions = sample.action.reshape(-1, 2)[owned]
        # Logistic shift of initial coordinates, also defined at 0 and 1.
        odds_multiplier = np.exp(-2 * np.clip(actions, -35., 35.))
        targets[owned] = positions / (positions + (1 - positions) * odds_multiplier)
    return samples, targets


def compare_deployments(pool, initial, learned, epochs, horizon):
    policies = [('initial', initial), ('learned', learned), ('hover', None), ('random', None)]
    jobs = [(points, epoch, horizon, label)
            for label, points in policies for epoch in epochs]
    return list(pool.map(evaluate_job, jobs))


def train(config):
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    raw = (ROOT / config['member_checkpoint']).read_bytes()
    source = torch.load(io.BytesIO(raw), weights_only=True, map_location='cpu')
    env = build_scene(settings_from_snapshot(source['environment']), episode=0)
    adapter = Adapter(env)
    observations, _, masks = adapter.master_inputs()
    # All actors receive the same team return, so share one central value.
    states = np.repeat(adapter.global_state()[None, :], adapter.masters, axis=0)
    inputs = observations, states, masks
    policy = Policy(adapter.master_obs_dim, adapter.global_dim, 2 * adapter.members,
                    True, config['hidden'])
    with torch.no_grad():
        policy.log_std.fill_(np.log(config['initial_std']))
    optimizer = torch.optim.Adam(policy.parameters(), lr=config['learning_rate'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / config['output_root'] / 'waypoint_mappo' / f"seed_{config['seed']}" / stamp
    output.mkdir(parents=True)
    print(output, flush=True)
    frozen = dict(environment=source['environment'], config=dict(hidden=source['config']['hidden']),
                  member_policies=source['member_policies'],
                  random_flight=dict(flight_seed=source['config']['flight_seed'],
                                     resample_slots=source['config']['resample_slots']))
    torch.save(frozen, output / 'frozen_members.pt')
    write_json(output / 'config.json', dict(**config, member_sha256=hashlib.sha256(raw).hexdigest(),
               member_episode=source['episodes_completed'], evaluation_slots=env.env_config.steps_per_epoch))

    def save_policy(name, iteration, targets):
        temporary = output / 'checkpoint.tmp'
        torch.save(dict(master=policy.state_dict(), targets=torch.from_numpy(targets),
                        iteration=iteration, hidden=config['hidden']), temporary)
        temporary.replace(output / name)

    initial = deployment(policy, inputs, True)[1]
    save_policy('initial.pt', 0, initial)
    with ProcessPoolExecutor(config['workers'], initializer=initialize_worker,
                             initargs=(output / 'frozen_members.pt',)) as pool:
        with (output / 'updates.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['update', 'epoch', 'sample_mean_delay_s', 'sample_min_delay_s',
                             'flight_violations', 'actor_change', 'kl'])
            for iteration in range(1, config['updates'] + 1):
                batch = [deployment(policy, inputs) for _ in range(config['deployments'])]
                epoch = config['train_epoch_start'] + iteration - 1
                jobs = [(targets, epoch, config['train_slots'], 'learned') for _, targets in batch]
                rows = list(pool.map(evaluate_job, jobs))
                delays = [row['mean_delay_s'] for row in rows]
                samples = []
                for (team, _), delay in zip(batch, delays):
                    for sample in team:
                        sample.reward = sample.target = -delay
                        sample.advantage = sample.target - sample.value
                    samples.extend(team)
                stats = update(policy, optimizer, samples, ppo_epochs=config['ppo_epochs'],
                               minibatch_size=len(samples), clip_ratio=.2, entropy_coef=0.,
                               value_coef=.5, max_grad_norm=.5)
                targets = deployment(policy, inputs, True)[1]
                save_policy('checkpoint.pt', iteration, targets)
                writer.writerow([iteration, epoch, np.mean(delays), min(delays),
                                 sum(r['flight_violations'] for r in rows), stats['actor_change'], stats['kl']])
                stream.flush()
                print(f"Update {iteration} | sampled delay {np.mean(delays):.4f}", flush=True)

        horizon = env.env_config.steps_per_epoch
        rows = compare_deployments(pool, initial, targets, config['test_epochs'], horizon)
        write_json(output / 'test_results.json', rows)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/methods/waypoint_mappo.toml')
    args = parser.parse_args()
    with args.config.open('rb') as stream:
        settings = tomllib.load(stream)['train']
    train(settings)
