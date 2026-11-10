"""CEM deployment baseline. Run with python -m methods.centralized_cem.run.

Optimizes normalized UAV destinations, not a general dynamic flight policy.
All objectives include travel from the original scene. Workload splits are
disjoint; final full-horizon test epochs are never used for selection.
CEM reference: https://www.mit.edu/~6.454/www_fall_2003/gew/CEtutorial.pdf
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
from time import perf_counter
import tomllib

import numpy as np
import torch

from methods.random_uav_ppo.support import settings_from_snapshot
from methods.frozen_member_ppo.run import frozen_policies
from methods.frozen_member_ppo.rollout import Adapter, collect_slot
from methods.frozen_member_ppo.support import build_scene, MetricSummary, write_json
from methods.random_uav_ppo.flight import RandomFlight

ROOT = Path(__file__).resolve().parents[2]


def destination_actions(positions, targets, side_length, full_step):
    # Saturation implements the physical per-axis speed constraint.
    return np.clip((targets * side_length - positions) / full_step, -1., 1.).astype(np.float32)


def cem_update(mean, std, candidates, costs, elite_count, smoothing, min_std):
    elite = candidates[np.argsort(costs)[:elite_count]]
    return ((1 - smoothing) * mean + smoothing * elite.mean(axis=0),
            np.maximum(min_std, (1 - smoothing) * std + smoothing * elite.std(axis=0)))


def initialize_worker(checkpoint):
    global ENV, ADAPTER, MEMBERS, FLIGHT
    torch.set_num_threads(1)
    source = torch.load(checkpoint, map_location='cpu', weights_only=True)
    ENV = build_scene(settings_from_snapshot(source['environment']), episode=0)
    ADAPTER = Adapter(ENV)
    _, MEMBERS = frozen_policies(ENV, source['config']['hidden'], source['member_policies'], 'cpu')
    FLIGHT = RandomFlight(ENV, **source['random_flight'])


def evaluate_job(job):
    targets, epoch, slots, label = job
    ENV.reset(epoch_index=epoch)
    summary = MetricSummary()
    count = ADAPTER.members
    full_step = ENV.members.config.max_horizontal_speed * ENV.env_config.flight_duration_s
    hover = np.zeros((count, 2), dtype=np.float32)
    with torch.inference_mode():
        for slot in range(slots):
            if label == 'random':
                actions = FLIGHT.actions(ENV, slot)
            elif label == 'hover':
                actions = hover
            else:
                actions = destination_actions(ENV.members.positions[:count, :2], targets,
                                              ENV.region.config.side_length, full_step)
            _, row = collect_slot(ADAPTER, None, MEMBERS, flight_actions=actions,
                                    boundary_penalty=1.)
            summary.add(row)
    return dict(policy=label, epoch=epoch, **summary.result())


def paired_result(learned, baseline):
    differences = np.array([b['mean_delay_s'] - a['mean_delay_s'] for a, b in zip(learned, baseline)])
    # Paired bootstrap across held-out episodes, not across correlated slots.
    rng = np.random.default_rng(2026)
    means = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
    return dict(delay_reduction_s=float(differences.mean()),
                relative_reduction=float(differences.mean() / np.mean([r['mean_delay_s'] for r in baseline])),
                paired_bootstrap_95_ci_s=np.quantile(means, [.025, .975]).tolist())


def train(config, resume_from=None):
    torch.set_num_threads(1)
    seed = config['seed']
    rng = np.random.default_rng(seed)
    source_path = (ROOT / config['member_checkpoint'] if resume_from is None
                   else Path(resume_from) / 'frozen_members.pt')
    raw = source_path.read_bytes()
    source = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    env = build_scene(settings_from_snapshot(source['environment']), episode=0)
    initial = env.members.positions[:env.members.member_uav_count, :2] / env.region.config.side_length
    horizon = env.env_config.steps_per_epoch
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / config['output_root'] / 'centralized_cem' / f'seed_{seed}' / stamp
    output.mkdir(parents=True)
    print(f'Output directory: {output}', flush=True)
    if resume_from is None:
        member_source = dict(path=str(source_path), sha256=hashlib.sha256(raw).hexdigest(),
                             episodes_completed=source['episodes_completed'])
        saved = dict(method='centralized_cem', environment=source['environment'],
                     config=dict(hidden=source['config']['hidden']), member_policies=source['member_policies'],
                     member_source=member_source, random_flight=dict(flight_seed=source['config']['flight_seed'],
                     resample_slots=source['config']['resample_slots']))
    else:
        saved = source
        member_source = saved['member_source']
    torch.save(saved, output / 'frozen_members.pt')
    write_json(output / 'config.json', dict(**config, member_source=member_source,
               resume_from=None if resume_from is None else str(Path(resume_from).resolve()),
               full_horizon=horizon, architecture='centralized static destinations, speed-limited travel, frozen argmax Members'))
    metadata = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                    generation=0, phase='search', output=str(output), pid=os.getpid())

    def status(**values):
        metadata.update(values)
        temporary = output / 'metadata.tmp'
        write_json(temporary, metadata)
        temporary.replace(output / 'metadata.json')

    mean = initial.copy()
    std = np.full_like(mean, config['initial_std'])
    best = initial.copy()
    best_cost = float('inf')
    selected = best.copy()
    selected_cost = float('inf')
    history, validation_history = [], []
    cache = {}
    start_generation = 0
    if resume_from is not None:
        previous = Path(resume_from)
        state = json.loads((previous / 'search_state.json').read_text(encoding='utf-8'))
        checkpoint = torch.load(previous / 'checkpoint.pt', map_location='cpu', weights_only=True)
        mean = np.array(state['mean'], dtype=np.float32)
        std = np.array(state['std'], dtype=np.float32)
        best = np.array(state['best'], dtype=np.float32)
        history, validation_history = state['history'], state['validation_history']
        rng.bit_generator.state = state['rng_state']
        best_cost = history[-1]
        selected = checkpoint['targets'].numpy().copy()
        selected_cost = min(validation_history)
        start_generation = checkpoint['generation']
        cache[best.tobytes()] = best_cost
        # Keep the previous selected policy usable even before the first new generation.
        torch.save(checkpoint, output / 'checkpoint.pt')
    status(generation=start_generation, resumed_from_generation=start_generation)
    started = perf_counter()
    plateau = False
    try:
        with ProcessPoolExecutor(max_workers=config['workers'], initializer=initialize_worker,
                                 initargs=(output / 'frozen_members.pt',)) as pool:
            def evaluate(targets, epochs, slots, label='learned'):
                return list(pool.map(evaluate_job, [(targets, e, slots, label) for e in epochs]))

            with (output / 'generations.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['generation', 'best_train_delay_s',
                    'population_mean_delay_s', 'search_std_mean', 'validation_delay_s', 'best_validation_delay_s', 'elapsed_s'])
                writer.writeheader()
                for generation in range(start_generation + 1, config['generations'] + 1):
                    candidates = np.clip(rng.normal(mean, std, size=(config['population'], *mean.shape)), .001, .999).astype(np.float32)
                    candidates[0], candidates[1] = best, mean
                    costs = np.empty(len(candidates))
                    pending = []
                    for index, target in enumerate(candidates):
                        key = target.tobytes()
                        if key in cache:
                            costs[index] = cache[key]
                        else:
                            pending.append((index, key))
                    jobs = [(candidates[i], epoch, config['train_slots'], 'learned')
                            for i, _ in pending for epoch in config['train_epochs']]
                    rows = list(pool.map(evaluate_job, jobs))
                    for offset, (index, key) in enumerate(pending):
                        n = len(config['train_epochs'])
                        costs[index] = np.mean([r['mean_delay_s'] for r in rows[offset*n:(offset+1)*n]])
                        cache[key] = float(costs[index])
                    winner = int(np.argmin(costs))
                    if costs[winner] < best_cost:
                        best, best_cost = candidates[winner].copy(), float(costs[winner])
                    mean, std = cem_update(mean, std, candidates, costs, config['elites'],
                                           config['smoothing'], config['min_std'])
                    history.append(best_cost)
                    validation = None
                    if generation == 1 or generation % config['validation_interval'] == 0:
                        status(generation=generation, phase='validation')
                        validation_rows = evaluate(best, config['validation_epochs'], horizon)
                        validation = float(np.mean([r['mean_delay_s'] for r in validation_rows]))
                        validation_history.append(validation)
                        if validation < selected_cost:
                            selected, selected_cost = best.copy(), validation
                        write_json(output / f'validation_{generation:03d}.json', validation_rows)
                    row = dict(generation=generation, best_train_delay_s=best_cost,
                        population_mean_delay_s=float(costs.mean()), search_std_mean=float(std.mean()),
                        validation_delay_s=validation, best_validation_delay_s=selected_cost,
                        elapsed_s=perf_counter() - started)
                    writer.writerow(row)
                    stream.flush()
                    torch.save(dict(**saved, targets=torch.from_numpy(selected.copy()),
                               train_targets=torch.from_numpy(best.copy()), generation=generation), output / 'checkpoint.tmp')
                    (output / 'checkpoint.tmp').replace(output / 'checkpoint.pt')
                    write_json(output / 'search_state.tmp', dict(mean=mean.tolist(), std=std.tolist(),
                        best=best.tolist(), history=history, validation_history=validation_history,
                        rng_state=rng.bit_generator.state))
                    (output / 'search_state.tmp').replace(output / 'search_state.json')
                    status(generation=generation, phase='search', best_train_delay_s=best_cost,
                           best_validation_delay_s=selected_cost, elapsed_s=row['elapsed_s'])
                    print(json.dumps(row), flush=True)
                    patience = config['patience']
                    if generation >= config['min_generations'] and validation is not None and len(validation_history) >= 3:
                        train_flat = (history[-patience] - best_cost) / history[-patience] < config['plateau_tolerance']
                        recent = validation_history[-3:]
                        validation_flat = (max(recent) - min(recent)) / min(recent) < config['plateau_tolerance']
                        plateau = train_flat and validation_flat
                        if plateau:
                            break
            status(phase='final_test')
            test = {label: evaluate(selected, config['test_epochs'], horizon, label)
                    for label in ('learned', 'hover', 'random')}
            comparisons = {label: paired_result(test['learned'], test[label]) for label in ('hover', 'random')}
            improves = all(r['relative_reduction'] >= .01 and r['paired_bootstrap_95_ci_s'][0] > 0
                           for r in comparisons.values())
            safe = all(np.mean([r['failure_rate'] for r in test['learned']]) <=
                       np.mean([r['failure_rate'] for r in test[label]]) for label in ('hover', 'random'))
            conclusion = 'converged_and_improved' if plateau and improves and safe else 'needs_followup'
            write_json(output / 'test_results.json', dict(episodes=test, comparisons=comparisons,
                       training_plateau=plateau, failure_not_worse=safe, conclusion=conclusion))
            status(status='completed', phase='done', conclusion=conclusion,
                   finished_at=datetime.now(timezone.utc).isoformat(), elapsed_s=perf_counter()-started)
    except BaseException as error:
        status(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/methods/centralized_cem.toml')
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--generations', type=int)
    parser.add_argument('--min-generations', type=int)
    parser.add_argument('--test-epoch-start', type=int)
    args = parser.parse_args()
    with args.config.open('rb') as stream:
        config = tomllib.load(stream)['train']
    if args.resume_from is not None:
        previous_config = json.loads((args.resume_from / 'config.json').read_text(encoding='utf-8'))
        config = {key: previous_config[key] for key in config}
    for key in ('generations', 'min_generations'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.test_epoch_start is not None:
        config['test_epochs'] = list(range(args.test_epoch_start, args.test_epoch_start + len(config['test_epochs'])))
    print(f'Completed: {train(config, args.resume_from)}', flush=True)
