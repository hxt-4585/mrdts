"""随机飞行 UAV 的配对评估：合法动作均匀随机、归属 Member 执行及多个 PPO 检查点。

python methods/random_uav_ppo/evaluate.py --checkpoints RUN/initial.pt RUN/checkpoint.pt --episodes 10
所有策略使用第一份检查点的环境快照和相同留出 DAG 流和随机飞行轨迹；不更新任何策略。
不同方法种子的检查点也可比较，空间场景由第一份快照统一指定。
方法种子默认来自第一份检查点；--seed 同时控制随机飞行和随机调度对照。
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

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from methods.random_uav_ppo.flight import derive_flight_seed
from methods.random_uav_ppo.rollout import Adapter, collect_slot
from methods.random_uav_ppo.run import load_checkpoint, output_directory, settings_from_snapshot
from methods.random_uav_ppo.support import MetricSummary
from methods.random_uav_ppo.support import build_scene, write_json


def digest_requests(digest, requests):
    """记录实际生成的拓扑、节点/边特征和归属，而非仅比较随机种子。"""
    for request in requests:
        dag = request.dag
        payload = [request.key, dag.node_num, dag.edges, dag.node_features.tolist(),
                   sorted(dag.edge_features.items())]
        digest.update(json.dumps(payload, separators=(',', ':')).encode('utf-8'))
        digest.update(b'\n')


def compare(checkpoints, *, episodes=10, epoch_start=1000000, seed=None,
            device='cpu', torch_threads=1, output_root='results', metrics_interval=10, resample_slots=None):
    torch.set_num_threads(torch_threads)
    _, members, source = load_checkpoint(checkpoints[0], device)
    flight_config = source['config'] if source['method'] == 'random_uav_ppo' else {}
    seed = source['config']['seed'] if seed is None else seed
    flight_seed = derive_flight_seed(seed)
    resample_slots = flight_config.get('resample_slots', 10) if resample_slots is None else resample_slots
    settings = settings_from_snapshot(source['environment'])
    output = output_directory(output_root, 'eval_random_uav', seed)
    policies = [('random', None), ('own', None)] + [
        (f'checkpoint_{index}', Path(path)) for index, path in enumerate(checkpoints)]
    write_json(output / 'config.json', dict(environment=source['environment'], episodes=episodes,
        epoch_start=epoch_start, seed=seed, metrics_interval=metrics_interval, flight_policy='random_vector', flight_seed=flight_seed, resample_slots=resample_slots, ordering='ERS',
        member_policy='checkpoint: argmax; random: uniform over current mask; own: owner Member',
        checkpoints={label: str(path.resolve()) for label, path in policies if path is not None},
        comparison='all policies use first checkpoint environment; paired episode/slot DAG streams and flight trajectories',
        reward_aggregation='per decision within Member; system delays weighted by DAG count'))
    summaries = {}
    started = perf_counter()
    with (output / 'episodes.csv').open('w', newline='', encoding='utf-8') as ep_file, \
            (output / 'members.csv').open('w', newline='', encoding='utf-8') as member_file, \
            (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as metric_file:
        ep_writer = member_writer = metric_writer = None
        for label, path in policies:
            if path is not None:
                _, members, _ = load_checkpoint(path, device)
            env = build_scene(settings, episode=epoch_start)
            adapter = Adapter(env, flight_seed=flight_seed, resample_slots=resample_slots)
            write_json(output / 'trajectory.json', adapter.flight.snapshot())
            total = MetricSummary()
            for episode in range(epoch_start, epoch_start + episodes):
                env.reset(epoch_index=episode)
                rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 102]))
                selector = None
                if label == 'random':
                    selector = lambda key, mask: int(rng.choice(np.flatnonzero(mask)))
                elif label == 'own':
                    selector = lambda key, mask: key.owner_member_id
                digest = hashlib.sha256()
                flight_digest = hashlib.sha256()
                counts = np.zeros(adapter.members, dtype=np.int64)
                rewards = np.zeros(adapter.members)
                actions = np.zeros((adapter.members, adapter.members + 2), dtype=np.int64)
                invalid = np.zeros(adapter.members, dtype=np.int64)
                summary = MetricSummary()
                for slot in range(env.env_config.steps_per_epoch):
                    low, row = collect_slot(adapter, members, deterministic=True,
                        member_selector=selector,
                        request_observer=lambda requests: digest_requests(digest, requests))
                    flight_digest.update(env.members.positions[:adapter.members, :2].tobytes())
                    for owner, samples in low.items():
                        counts[owner] += len(samples)
                        for sample in samples:
                            action = int(sample.action)
                            rewards[owner] += sample.reward
                            actions[owner, action] += 1
                            invalid[owner] += not sample.mask[action]
                    summary.add(row)
                    total.add(row)
                    row = dict(policy=label, **row)
                    if metric_writer is None:
                        metric_writer = csv.DictWriter(metric_file, fieldnames=list(row))
                        metric_writer.writeheader()
                    if (slot + 1) % metrics_interval == 0 or slot + 1 == env.env_config.steps_per_epoch:
                        metric_writer.writerow(row)
                row = dict(policy=label, episode=episode, dag_stream_sha256=digest.hexdigest(),
                           flight_stream_sha256=flight_digest.hexdigest(),
                           **summary.result())
                if ep_writer is None:
                    ep_writer = csv.DictWriter(ep_file, fieldnames=list(row))
                    ep_writer.writeheader()
                ep_writer.writerow(row)
                for owner, count in enumerate(counts):
                    row = dict(policy=label, episode=episode, member=owner, samples=int(count),
                        reward_sum=float(rewards[owner]), reward_mean=float(rewards[owner] / count) if count else None,
                        self_fraction=float(actions[owner, owner] / count) if count else None,
                        invalid_actions=int(invalid[owner]))
                    row.update({f'action_{i}_fraction': float(value / count) if count else None
                                for i, value in enumerate(actions[owner])})
                    if member_writer is None:
                        member_writer = csv.DictWriter(member_file, fieldnames=list(row))
                        member_writer.writeheader()
                    member_writer.writerow(row)
                ep_file.flush()
                member_file.flush()
                metric_file.flush()
                print(f'{label} | held-out episode {episode} | '
                      f'delay {summary.result()["mean_delay_s"]:.4f}s | '
                      f'failure {summary.result()["failure_rate"]:.4f}', flush=True)
            summaries[label] = total.result()
    write_json(output / 'summary.json', dict(policies=summaries, elapsed_s=perf_counter() - started))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--epoch-start', type=int, default=1000000)
    parser.add_argument('--seed', type=int, help='方法种子；默认读取第一份检查点，统一派生飞行与随机调度子流')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--metrics-interval', type=int, default=10)
    parser.add_argument('--resample-slots', type=int)
    parser.add_argument('--output-root', default='results')
    print(f'Completed: {compare(**vars(parser.parse_args()))}', flush=True)


if __name__ == '__main__':
    main()
