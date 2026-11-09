"""同一冻结低层下，比较学习飞行、随机飞行和悬停；使用留出 DAG 流。

python methods/frozen_member_ppo/evaluate.py --checkpoint RUN/checkpoint.pt --episodes 10
各方法使用相同环境及 epoch；飞行轨迹和任务归属随飞行策略变化。
"""

import argparse
import csv
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from methods.frozen_member_ppo.run import load_checkpoint, output_directory
from methods.hmappo.rollout import Adapter, collect_slot
from methods.hmappo.support import MetricSummary, write_json
from methods.random_uav_ppo.flight import RandomFlight


def compare(checkpoint, *, episodes=10, epoch_start=1000000, device='cpu',
            torch_threads=1, output_root='results'):
    torch.set_num_threads(torch_threads)
    env, master, members, saved = load_checkpoint(checkpoint, device)
    adapter = Adapter(env)
    flight = RandomFlight(env, **saved['random_flight'])
    hover = np.zeros((adapter.members, 2), dtype=np.float32)
    output = output_directory(output_root, 'eval', saved['config']['seed'])
    write_json(output / 'config.json', dict(checkpoint=str(Path(checkpoint).resolve()),
        episodes=episodes, epoch_start=epoch_start, environment=saved['environment'],
        member_source=saved['member_source'], member_policy='frozen argmax',
        random_flight=saved['random_flight'], boundary_penalty=saved['config']['boundary_penalty'],
        comparison='same scene and workload epochs; trajectories and ownership differ by flight policy'))
    summaries = {}
    started = perf_counter()
    with (output / 'episodes.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = None
        for label in ('learned', 'random', 'hover'):
            total = MetricSummary()
            for episode in range(epoch_start, epoch_start + episodes):
                env.reset(epoch_index=episode)
                summary = MetricSummary()
                for slot in range(env.env_config.steps_per_epoch):
                    actions = None
                    if label == 'random':
                        actions = flight.actions(env, slot)
                    elif label == 'hover':
                        actions = hover
                    _, _, row = collect_slot(adapter, master, members, deterministic=True,
                        member_deterministic=True, flight_actions=actions,
                        boundary_penalty=saved['config']['boundary_penalty'])
                    summary.add(row)
                    total.add(row)
                row = dict(policy=label, episode=episode, **summary.result())
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                print(f'{label} | Episode {episode} | Delay {row["mean_delay_s"]:.4f}s | '
                      f'Failure {row["failure_rate"]:.4f}', flush=True)
            summaries[label] = total.result()
    write_json(output / 'summary.json', dict(policies=summaries, elapsed_s=perf_counter() - started))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--epoch-start', type=int, default=1000000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--output-root', default='results')
    print(f'Completed: {compare(**vars(parser.parse_args()))}', flush=True)
