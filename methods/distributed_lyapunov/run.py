"""python -m methods.distributed_lyapunov.run --output results/distributed_lyapunov/RUN

Background experiment: pilot feasibility/V search, two training routes with
validation early stopping, then independent continuous-horizon queue tests.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

from .learning import MemberPolicy, update
from .planner import Controller, frames
from .simulation import build_env, execute, queue_statistics


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def save(path, policies, controller, config, env, episode):
    temp = path.with_suffix('.tmp')
    torch.save(dict(policies=policies.state_dict(), controller=controller, config=config,
                    environment=env.settings, episode=episode), temp)
    temp.replace(path)


def episode(env, policies, controller, *, epoch, slots, v, random_flight=False,
            train_low=False, train_high=False, optimizers=None, exploration=0.,
            heuristic=False, stream=None, update_slots=8):
    env.reset(epoch_index=epoch)
    rng = np.random.default_rng(np.random.SeedSequence([45, epoch, 102]))
    buffer = [[] for _ in policies]
    rows, losses = [], []
    start = time.perf_counter()
    action = np.zeros((env.members.member_uav_count, 2), np.float32)
    for slot in range(slots):
        state = frames(env)
        if random_flight:
            if slot % 10 == 0:
                action = rng.uniform(-1., 1., action.shape).astype(np.float32)
            xy = env.members.positions[:len(action), :2]
            step = env.members.config.max_horizontal_speed * env.env_config.flight_duration_s
            predicted = xy + action * step
            action = np.where((predicted < 0.) | (predicted > env.region.config.side_length), -action, action)
            payload = 0
        else:
            action, payload = controller.actions(state, exploration)
        samples, row, costs = execute(env, policies, action, v,
                                     deterministic=not train_low, heuristic=heuristic)
        row.update(slot=slot, epoch=epoch, neighbor_payload_scalars=payload)
        rows.append(row)
        if stream is not None:
            stream.write(json.dumps(row) + '\n')
            stream.flush()
        if train_low:
            for batch, records in zip(buffer, samples):
                batch.extend(records)
            if (slot + 1) % update_slots == 0 or slot + 1 == slots:
                losses.extend(update(p, opt, batch) for p, opt, batch in zip(policies, optimizers, buffer))
                buffer = [[] for _ in policies]
        if train_high:
            controller.feedback(state, action, costs, v)
    stats = queue_statistics([r['queue_j'] for r in rows], [r['energy_j'] for r in rows],
                             env.members.config.member_slot_energy)
    stats.update(mean_delay_s=float(np.mean([r['mean_delay_s'] for r in rows])),
                 normalized_cost=float(np.mean([r['objective'] / (v * env.users.num_users) for r in rows])),
                 failure_rate=float(np.mean([r['failure_rate'] for r in rows])),
                 wall_seconds=time.perf_counter()-start, slots=slots, epoch=epoch,
                 ppo_loss=float(np.mean(losses)) if losses else None)
    return stats


def validation(env, policies, controller, config, phase, epochs, slots):
    # A separate environment preserves the training RNG and current queues.
    from env.env import MRDTSEnv
    other = MRDTSEnv(env.settings)
    other.generate_scene()
    return [episode(other, policies, controller, epoch=e, slots=slots, v=config['v'],
                    random_flight=phase == 'low') for e in epochs]


def ranking(results):
    # Prefer per-UAV empirical stability over a smaller delay/cost alone.
    unstable = sum(not r['empirically_stable'] for r in results)
    return (unstable,
            max(0., max(max(r['tail_slope_j_per_slot']) for r in results)) if unstable else 0.,
            float(np.mean([r['normalized_cost'] for r in results])))


def train(config, output, mode, low_checkpoint=None):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    env = build_env(config['budget'])
    policies = nn.ModuleList([MemberPolicy(env.members.member_uav_count) for _ in range(env.members.member_uav_count)])
    optimizers = [torch.optim.Adam(p.parameters(), lr=.0003) for p in policies]
    controller = Controller(frames(env), config['seed'])
    if low_checkpoint is not None:
        saved = torch.load(low_checkpoint, weights_only=False, map_location='cpu')
        policies.load_state_dict(saved['policies'])
        controller = saved['controller']
        from env.env import MRDTSEnv
        env = MRDTSEnv(saved['environment'])
        env.generate_scene()
    write_json(output / 'config.json', dict(config, mode=mode, low_checkpoint=str(low_checkpoint),
               environment={k: asdict(v) for k, v in env.settings.items()}))
    if low_checkpoint is None:
        save(output / 'initial.pt', policies, controller, config, env, 0)
    phases = ['high'] if low_checkpoint is not None else ['joint'] if mode == 'joint' else ['low', 'high']
    history = (json.loads((output/'history.json').read_text(encoding='utf-8'))
               if low_checkpoint is not None and (output/'history.json').exists() else [])
    write_json(output / 'status.json', dict(state='training', phase=phases[0], episode=0,
               pid=os.getpid(), low_checkpoint=str(low_checkpoint)))
    for phase in phases:
        best_rank, stale = None, 0
        maximum = config['low_max_episodes'] if phase == 'low' else config['high_episodes']
        for index in range(maximum):
            epoch = 400000 + (0 if phase != 'high' else 100000) + index
            with (output / 'slots.jsonl').open('a', encoding='utf-8') as stream:
                result = episode(env, policies, controller, epoch=epoch, slots=config['train_slots'],
                         v=config['v'], random_flight=phase=='low', train_low=phase!='high',
                         train_high=phase!='low', optimizers=optimizers,
                         exploration=max(.1, .8*(1-index/maximum)), stream=stream)
            result.update(phase=phase, episode=index+1)
            history.append(result)
            write_json(output / 'history.json', history)
            save(output / 'latest.pt', policies, controller, config, env, index+1)
            write_json(output / 'status.json', dict(state='training', phase=phase, episode=index+1,
                       pid=os.getpid(), latest=result))
            print(json.dumps(dict(mode=mode, phase=phase, episode=index+1, delay=result['mean_delay_s'],
                                  max_queue=max(result['final_queue_j']), seconds=result['wall_seconds'])), flush=True)
            if (index+1) % config['validation_interval'] == 0:
                results = validation(env, policies, controller, config, phase, [1500000,1500001],
                                     config['validation_slots'])
                score = ranking(results)
                # Slope tolerance prevents tiny fluctuations consuming patience.
                compare = (score[0], round(score[1], 1), round(score[2], 2))
                improved = best_rank is None or compare < best_rank
                with (output / 'validation.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(phase=phase, episode=index+1, results=results,
                                                 ranking=compare, improved=improved))+'\n')
                if improved:
                    best_rank, stale = compare, 0
                    save(output / f'{phase}_best.pt', policies, controller, config, env, index+1)
                else:
                    stale += 1
                if index+1 >= config['min_episodes'] and stale >= config['patience']:
                    break
        selected = torch.load(output / f'{phase}_best.pt', weights_only=False, map_location='cpu')
        policies.load_state_dict(selected['policies'])
        controller = selected['controller']
        save(output / 'selected.pt', policies, controller, config, env, selected['episode'])
    write_json(output / 'status.json', dict(state='continuous_evaluation', pid=os.getpid()))
    results = []
    for epoch in [2500000,2500001,2500002]:
        with (output / f'stability_{epoch}.jsonl').open('w', encoding='utf-8') as stream:
            results.append(episode(env, policies, controller, epoch=epoch,
                           slots=config['test_slots'], v=config['v'], stream=stream))
        write_json(output / 'stability.json', results)
    write_json(output / 'status.json', dict(state='complete', mode=mode, pid=os.getpid(),
               empirically_stable=all(r['empirically_stable'] for r in results),
               results=results))


def pilot_needs_confirmation(result, budget):
    """Short noisy windows screen energy headroom, not queue stationarity."""
    return result['empirically_stable'] or max(result['mean_energy_j']) <= .99 * budget


def pilot(output, config):
    torch.set_num_threads(1)
    saved = output / 'pilot.json'
    all_results = json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else []
    confirmed_path = output / 'pilot_confirmation.json'
    confirmations = (json.loads(confirmed_path.read_text(encoding='utf-8'))
                     if confirmed_path.exists() else [])
    for budget in [145., 170., 195., 220.]:
        candidates = []
        for v in [1000., 10000., 100000.]:
            result = next((r for r in all_results if r['budget'] == budget and r['v'] == v), None)
            if result is None:
                env = build_env(budget)
                policies = [None] * env.members.member_uav_count
                result = episode(env, policies, None, epoch=3000000, slots=config['pilot_slots'], v=v,
                                 random_flight=True, heuristic=True)
                result.update(budget=budget, v=v)
                all_results.append(result)
                write_json(saved, all_results)
            if pilot_needs_confirmation(result, budget):
                confirmed = next((r for r in confirmations if r['budget'] == budget and r['v'] == v), None)
                if confirmed is None:
                    env = build_env(budget)
                    with (output / f'pilot_confirmation_{int(budget)}_{int(v)}.jsonl').open('w', encoding='utf-8') as stream:
                        confirmed = episode(env, [None]*env.members.member_uav_count, None,
                                    epoch=3000000, slots=config['pilot_confirmation_slots'], v=v,
                                    random_flight=True, heuristic=True, stream=stream)
                    confirmed.update(budget=budget, v=v)
                    confirmations.append(confirmed)
                    write_json(confirmed_path, confirmations)
                result = confirmed
            candidates.append(result)
        feasible = [r for r in candidates if r['empirically_stable']]
        if feasible:
            selected = min(feasible, key=lambda r: r['mean_delay_s'])
            config.update(budget=budget, v=selected['v'])
            write_json(output / 'selection.json', dict(config, pilot_basis=selected,
                       caveat='Pilot is a feasibility screen, not evidence that trained policies are stable.'))
            return
    raise RuntimeError('No empirically feasible pilot within 145..220 J; inspect pilot.json before extending budget.')


def experiment(output, config):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'status.json', dict(state='pilot', pid=os.getpid(), config=config))
    pilot(output, config)
    workers = []
    for mode in ['joint','staged']:
        destination = output / mode
        destination.mkdir(exist_ok=True)
        with (destination / 'stdout.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'methods.distributed_lyapunov.run',
                         '--output', str(destination), '--mode', mode, '--selection', str(output/'selection.json')],
                         stdout=log, stderr=subprocess.STDOUT)
        workers.append(dict(mode=mode, pid=process.pid, output=str(destination)))
    write_json(output / 'status.json', dict(state='dispatched', pid=os.getpid(), workers=workers,
               config=config))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['experiment','joint','staged','smoke'], default='experiment')
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--low-checkpoint', type=Path)
    args = parser.parse_args()
    config = dict(seed=45, budget=145., v=10000., pilot_slots=128, pilot_confirmation_slots=1024, train_slots=500,
                  validation_slots=500, validation_interval=5, min_episodes=15, patience=3,
                  low_max_episodes=80, high_episodes=60, test_slots=3000)
    if args.selection:
        config.update({k:v for k,v in json.loads(args.selection.read_text(encoding='utf-8')).items() if k in config})
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == 'experiment':
            experiment(args.output, config)
        elif args.mode == 'smoke':
            config.update(train_slots=3, validation_slots=3, validation_interval=1,
                          min_episodes=1, patience=1, low_max_episodes=1, high_episodes=1, test_slots=4)
            train(config, args.output / 'staged', 'staged')
            train(config, args.output / 'joint', 'joint')
            env = build_env(145.)
            episode(env, [None]*env.members.member_uav_count, None, epoch=3, slots=2,
                    v=1000., random_flight=True, heuristic=True)
        else:
            train(config, args.output, args.mode, args.low_checkpoint)
    except Exception as error:
        write_json(args.output / 'status.json', dict(state='failed', error=repr(error), pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()
