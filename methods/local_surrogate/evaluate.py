"""在留出 DAG 流比较局部搜索、随机飞行、悬停；低层和代理模型均冻结。

python -m methods.local_surrogate.evaluate --checkpoint RUN/checkpoint.pt
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

from methods.random_uav_ppo.support import settings_from_snapshot
from methods.frozen_member_ppo.run import frozen_policies
from methods.hmappo.rollout import Adapter
from methods.hmappo.support import build_scene, write_json
from methods.local_surrogate.planner import LocalPlanner, capture_views
from methods.local_surrogate.rollout import Summary, execute_slot, regional_delays, lyapunov_cost
from methods.local_surrogate.run import output_directory, planner_options
from methods.local_surrogate.surrogate import OnlineSurrogate
from methods.random_uav_ppo.flight import RandomFlight


def compare(checkpoint, *, episodes=5, slots=None, epoch_start=1000000, device='cpu',
            torch_threads=1, output_root='results'):
    torch.set_num_threads(torch_threads)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    env = build_scene(settings_from_snapshot(saved['environment']), episode=epoch_start)
    adapter = Adapter(env)
    unused_master, members = frozen_policies(env, saved['config']['hidden'], saved['member_policies'], device)
    del unused_master
    planner = LocalPlanner(env, **planner_options(saved['config']))
    planner.models = [OnlineSurrogate.from_state(state) for state in saved['surrogates']]
    flight = RandomFlight(env, **saved['random_flight'])
    output = output_directory(output_root, 'eval', saved['config']['seed'])
    write_json(output / 'config.json', dict(checkpoint=str(Path(checkpoint).resolve()),
        episodes=episodes, slots=slots, epoch_start=epoch_start, environment=saved['environment'],
        member_source=saved['member_source'], random_flight=saved['random_flight'],
        objective='sum(Q/E_budget * E/E_budget) + delay_weight * sum(D/H)',
        evaluation='same scene/workload epochs; frozen surrogate and low layer; no online adaptation',
        timing='single-process planning and simulation measured separately',
        information='extended one-hop telemetry; not identical to original position-only HMAPPO inputs'))
    summaries = {}
    started = perf_counter()
    horizon = env.env_config.steps_per_epoch if slots is None else slots
    with (output / 'slots.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = None
        for label in ('local_surrogate', 'random', 'hover'):
            total = Summary()
            for episode in range(epoch_start, epoch_start + episodes):
                env.reset(epoch_index=episode)
                planner.reset(episode)
                last_delays = np.zeros(adapter.masters)
                for slot in range(horizon):
                    planning_started = perf_counter()
                    views = capture_views(env, last_delays)
                    stats = dict(candidate_evaluations=0, quote_messages=0, plan_messages=0,
                                 neighbor_messages=0, neighbor_payload_scalars=0)
                    if label == 'local_surrogate':
                        actions, stats = planner.actions(views)
                    elif label == 'random':
                        actions = flight.actions(env, slot)
                    else:
                        actions = np.zeros((adapter.members, 2))
                    planning_s = perf_counter() - planning_started
                    simulation_started = perf_counter()
                    result, row = execute_slot(adapter, members, actions)
                    simulation_s = perf_counter() - simulation_started
                    last_delays = regional_delays(result, env.users.region_ids, adapter.masters,
                                                  env.env_config.scheduling_duration_s)
                    row.update(policy=label, planning_s=planning_s, simulation_s=simulation_s,
                        model_update_s=0., lyapunov_cost=lyapunov_cost(
                            views, result, last_delays, planner.energy_budget, planner.delay_weight), **stats)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    total.add(row)
                stream.flush()
                print(f'{label} | Episode {episode} | slots {horizon}', flush=True)
            summaries[label] = total.result()
    frozen = all(all(torch.equal(value, state[key]) if isinstance(value, torch.Tensor)
                     else value == state[key] for key, value in model.state().items())
                 for model, state in zip(planner.models, saved['surrogates']))
    write_json(output / 'summary.json', dict(policies=summaries, surrogates_frozen=frozen,
        training_slots=saved['slots_completed'], elapsed_s=perf_counter() - started))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--slots', type=int)
    parser.add_argument('--epoch-start', type=int, default=1000000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--output-root', default='results')
    print(f'Completed: {compare(**vars(parser.parse_args()))}', flush=True)
