"""局部代理搜索高层 + 冻结的预训练 Member PPO。

python -m methods.local_surrogate.run --config config/methods/local_surrogate.toml
不调用全局仿真器评价候选；只从实际执行的时隙学习。
共享带宽/BS 的远端影响没有精确建模，不宣称严格求解 P2 或能量稳定性保证。
"""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import sys
from time import perf_counter
import tomllib
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from methods.random_uav_ppo.support import settings_from_snapshot
from methods.frozen_member_ppo.run import frozen_policies
from methods.hmappo.rollout import Adapter
from methods.hmappo.support import build_scene, write_json
from methods.local_surrogate.planner import LocalPlanner, capture_views
from methods.local_surrogate.rollout import (Summary, execute_slot, regional_delays,
                                            physical_target, lyapunov_cost)


def output_directory(root, mode, seed):
    root = Path(root)
    if not root.is_absolute():
        root = ROOT / root
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
    output = root / 'local_surrogate' / f'seed_{seed}' / f'{mode}_{stamp}'
    output.mkdir(parents=True)
    print(f'Output directory: {output}', flush=True)
    return output


def planner_options(config):
    return {key: config[key] for key in ('candidates', 'rounds', 'features', 'ridge', 'seed',
                                        'delay_weight', 'uncertainty_penalty')}


def train(*, member_checkpoint, episodes=30, slots=None, seed=45, device='cpu', torch_threads=1,
          warmup_slots=128, exploration=.15, candidates=32, rounds=2, features=32, ridge=1.,
          delay_weight=1., uncertainty_penalty=.05, log_interval=50, output_root='results'):
    config = dict(member_checkpoint=str(member_checkpoint), episodes=episodes, slots=slots,
        seed=seed, device=device, torch_threads=torch_threads, warmup_slots=warmup_slots,
        exploration=exploration, candidates=candidates, rounds=rounds, features=features,
        ridge=ridge, delay_weight=delay_weight, uncertainty_penalty=uncertainty_penalty,
        log_interval=log_interval, output_root=str(output_root))
    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    path = Path(member_checkpoint)
    if not path.is_absolute():
        path = ROOT / path
    raw = path.read_bytes()
    source = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    member_source = dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                         method=source['method'], episodes_completed=source['episodes_completed'])
    settings = settings_from_snapshot(source['environment'])
    if slots is not None:
        settings['env'] = replace(settings['env'], steps_per_epoch=slots)
    env = build_scene(settings, episode=0)
    adapter = Adapter(env)
    unused_master, members = frozen_policies(env, source['config']['hidden'], source['member_policies'], device)
    del unused_master
    hidden = source['config']['hidden']
    flight_config = dict(flight_seed=source['config']['flight_seed'],
                         resample_slots=source['config']['resample_slots'])
    del source, raw
    planner = LocalPlanner(env, **planner_options(config))
    snapshot = {name: asdict(value) for name, value in settings.items()}
    output = output_directory(output_root, 'train', seed)
    write_json(output / 'config.json', dict(method='local_surrogate', training=config,
        environment=snapshot, member_source=member_source, frozen_members=True,
        member_policy='masked argmax; ERS; original delay-only pretrained policy',
        objective='sum(Q/E_budget * E/E_budget) + delay_weight * sum(D/H)',
        information='own + one-hop positions, queues, fixed GD positions, previous regional delay',
        coordination='sequential block search; neighbors return scalar costs for candidate plans',
        execution='single-process protocol simulation; no parallel speedup claim',
        limitations='local approximation; unmodelled distant bandwidth/BS effects; no stability guarantee',
        timing='planning_s includes view construction and candidate search; model updates measured separately',
        communication='request/reply batches + selected-plan broadcasts; base telemetry and headers excluded',
        evaluation_epoch_start=1000000))
    metadata = dict(status='running', started_at=datetime.now(timezone.utc).isoformat())
    write_json(output / 'metadata.json', metadata)
    summary = Summary()
    started = perf_counter()

    def save(completed):
        saved = dict(method='local_surrogate', config=dict(config, hidden=hidden), environment=snapshot,
            member_source=member_source, member_policies=members.state_dict(), random_flight=flight_config,
            episodes_completed=completed, slots_completed=summary.slots,
            surrogates=[model.state() for model in planner.models])
        temporary = output / 'checkpoint.tmp'
        torch.save(saved, temporary)
        temporary.replace(output / 'checkpoint.pt')

    try:
        with (output / 'slots.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = None
            for episode in range(episodes):
                env.reset(epoch_index=episode)
                planner.reset(episode)
                last_delays = np.zeros(adapter.masters)
                episode_summary = Summary()
                for slot in range(env.env_config.steps_per_epoch):
                    planning_started = perf_counter()
                    views = capture_views(env, last_delays)
                    explore = 1. if summary.slots < warmup_slots else exploration
                    actions, stats = planner.actions(views, explore=explore)
                    planning_s = perf_counter() - planning_started
                    simulation_started = perf_counter()
                    result, row = execute_slot(adapter, members, actions)
                    simulation_s = perf_counter() - simulation_started
                    last_delays = regional_delays(result, env.users.region_ids, adapter.masters,
                                                  env.env_config.scheduling_duration_s)
                    cost = lyapunov_cost(views, result, last_delays, planner.energy_budget, delay_weight)
                    update_started = perf_counter()
                    residuals = []
                    for owner, (view, model) in enumerate(zip(views, planner.models)):
                        target = physical_target(view, result, last_delays[planner.regions[owner] - 1],
                                                 planner.energy_budget)
                        residual = model.update(view.features(actions), target)
                        residuals.extend(residual[np.concatenate(([True], view.owned))])
                    row.update(lyapunov_cost=cost, exploration=explore, planning_s=planning_s,
                        simulation_s=simulation_s, model_update_s=perf_counter() - update_started,
                        prediction_rmse=float(np.sqrt(np.mean(np.square(residuals)))), **stats)
                    summary.add(row)
                    episode_summary.add(row)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    if summary.slots % log_interval == 0:
                        save(episode)
                        write_json(output / 'metadata.json', dict(metadata, slots_completed=summary.slots,
                                   episode=episode, slot=slot + 1, elapsed_s=perf_counter() - started))
                        print(f'Episode {episode + 1} slot {slot + 1} | delay {row["mean_delay_s"]:.4f}s | '
                              f'cost {cost:.3f} | plan {planning_s:.4f}s', flush=True)
                save(episode + 1)
                print(f'Episode {episode + 1}/{episodes} | {episode_summary.result()}', flush=True)
        write_json(output / 'summary.json', dict(**summary.result(), elapsed_s=perf_counter() - started))
        metadata['status'] = 'completed'
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        metadata.update(finished_at=datetime.now(timezone.utc).isoformat(), slots_completed=summary.slots,
                        elapsed_s=perf_counter() - started)
        write_json(output / 'metadata.json', metadata)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/methods/local_surrogate.toml')
    parser.add_argument('--member-checkpoint', type=Path)
    for name in ('episodes', 'slots', 'seed', 'torch-threads', 'warmup-slots', 'candidates',
                 'rounds', 'features', 'log-interval'):
        parser.add_argument('--' + name, type=int)
    for name in ('exploration', 'ridge', 'delay-weight', 'uncertainty-penalty'):
        parser.add_argument('--' + name, type=float)
    parser.add_argument('--device')
    parser.add_argument('--output-root')
    args = parser.parse_args()
    with args.config.open('rb') as stream:
        config = tomllib.load(stream)['train']
    config.update({key: value for key, value in vars(args).items() if key != 'config' and value is not None})
    print(f'Completed: {train(**config)}', flush=True)


if __name__ == '__main__':
    main()
