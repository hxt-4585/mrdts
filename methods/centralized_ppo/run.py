"""Train a centralized terminal-action PPO policy, independently of other methods.

python -B -m methods.centralized_ppo.run
python -B -m methods.centralized_ppo.run --resume-from PATH --updates 300
Use --skip-test for provisional runs; select extensions using validation only.

One episode is one joint deployment action and its negative mean DAG delay.
There is no temporal bootstrapping, elite selection, CEM warm start, or teacher.
Mirrored samples share a workload stream; deterministic neural-policy outputs
are selected using full-horizon validation. Test workloads are used only after
training. Evaluation budgets are logged separately from environment slots.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
import tomllib

import numpy as np
import torch

from methods.centralized_ppo.policy import JointPPO
from methods.centralized_ppo.simulation import Simulator, initialize_worker, evaluate_job

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, content):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, content):
    temporary = path.with_suffix('.tmp')
    torch.save(content, temporary)
    temporary.replace(path)


def exploration_std(config, iteration):
    fraction = min((iteration - 1) / config['anneal_updates'], 1.)
    return config['initial_std'] * (config['final_std'] / config['initial_std']) ** fraction


def paired_comparison(learned, reference):
    # Positive differences mean PPO has higher delay. Resample entire episodes.
    a = np.array([row['mean_delay_s'] for row in learned])
    b = np.array([row['mean_delay_s'] for row in reference])
    delta = a - b
    random = np.random.default_rng(2026)
    indices = random.integers(len(delta), size=(10000, len(delta)))
    samples = delta[indices].mean(axis=1)
    interval = np.quantile(samples, [.025, .975])
    margin_interval = np.quantile((a - 1.01 * b)[indices].mean(axis=1), [.025, .975])
    failures_ok = sum(row['failed_dags'] for row in learned) <= sum(row['failed_dags'] for row in reference)
    return dict(learned_mean_delay_s=float(a.mean()), reference_mean_delay_s=float(b.mean()),
                delay_difference_s=float(delta.mean()), relative_difference=float(delta.mean() / b.mean()),
                paired_bootstrap_95_ci_s=interval.tolist(), failure_not_worse=failures_ok,
                paired_1pct_margin_95_ci_s=margin_interval.tolist(),
                point_estimate_within_1pct=bool(a.mean() <= b.mean() * 1.01 and failures_ok),
                bootstrap_upper_within_1pct=bool(margin_interval[1] <= 0 and failures_ok),
                bootstrap_upper_below_zero=bool(interval[1] < 0 and failures_ok))


def final_evaluation(pool, output, config, report_dir=None):
    # CEM coordinates first enter this baseline here, after selection has ended.
    best = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    simulator = Simulator(torch.load(output / 'frozen_members.pt', map_location='cpu', weights_only=True))
    selected_actor = JointPPO(simulator.observation, simulator.initial, config['hidden'], config['learning_rate'])
    selected_actor.load_state_dict(best['policy'])
    cem = torch.load(ROOT / config['cem_reference'], map_location='cpu', weights_only=True)
    candidates = {'learned': selected_actor.targets(), 'cem': cem['targets'].numpy(),
                  'hover': None, 'random': None}
    report_dir = output if report_dir is None else report_dir
    episodes = {}
    for label, targets in candidates.items():
        jobs = [(targets, epoch, config['evaluation_slots'], label) for epoch in config['test_epochs']]
        episodes[label] = list(pool.map(evaluate_job, jobs))
        write_json(report_dir / f'test_{label}.json', episodes[label])
    comparisons = {label: paired_comparison(episodes['learned'], episodes[label])
                   for label in ('cem', 'hover', 'random')}
    write_json(report_dir / 'test_results.json', dict(episodes=episodes, comparisons=comparisons,
               selected_iteration=best['iteration'], validation_delay_s=best['best_validation_delay_s'],
               cem_reference=config['cem_reference'],
               cem_checkpoint_sha256=hashlib.sha256((ROOT / config['cem_reference']).read_bytes()).hexdigest(),
               interpretation='Fixed spatial scene, held-out workloads; budgets differ from CEM.'))
    return comparisons


def train(config, *, output_dir=None, resume_from=None):
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = (ROOT / config['output_root'] / 'centralized_ppo' / f"seed_{config['seed']}" / stamp
              if output_dir is None else Path(output_dir))
    output.mkdir(parents=True)
    started = perf_counter()
    if resume_from is None:
        member_path = ROOT / config['member_checkpoint']
        source = torch.load(member_path, map_location='cpu', weights_only=True)
        digest = hashlib.sha256(member_path.read_bytes()).hexdigest()
        frozen = {key: source[key] for key in ('environment', 'member_policies', 'episodes_completed')}
        frozen['config'] = {key: source['config'][key] for key in ('hidden', 'flight_seed', 'resample_slots')}
        save_checkpoint(output / 'frozen_members.pt', frozen)
    else:
        resume_from = Path(resume_from)
        shutil.copyfile(resume_from / 'frozen_members.pt', output / 'frozen_members.pt')
        frozen = torch.load(output / 'frozen_members.pt', map_location='cpu', weights_only=True)
        previous_config = json.loads((resume_from / 'config.json').read_text(encoding='utf-8'))
        digest = previous_config['member_sha256']
    simulator = Simulator(frozen)
    agent = JointPPO(simulator.observation, simulator.initial, config['hidden'], config['learning_rate'])
    write_json(output / 'config.json', dict(config, member_sha256=digest,
        member_episode=frozen['episodes_completed'], environment=frozen['environment'],
        action='one centralized joint deployment vector; deterministic sigmoid mean at evaluation',
        algorithm='terminal-action PPO, joint log probability, mirrored sampling, exact KL backtracking',
        resume_from=None if resume_from is None else str(resume_from.resolve())))
    metadata = dict(status='running', phase='initial_validation', pid=os.getpid(), iteration=0,
                    observation_dim=len(simulator.observation),
                    output=str(output.resolve()), started_at=datetime.now(timezone.utc).isoformat(),
                    train_slots_evaluated=0, validation_slots_evaluated=0, test_slots_evaluated=0)
    best_delay = float('inf')
    first_iteration = 1
    elapsed_before = 0.
    if resume_from is not None:
        saved = torch.load(resume_from / 'latest.pt', map_location='cpu', weights_only=True)
        agent.load_state_dict(saved['policy'])
        agent.optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        best_delay = saved['best_validation_delay_s']
        first_iteration = saved['iteration'] + 1
        elapsed_before = saved['elapsed_s']
        metadata.update({key: saved[key] for key in ('train_slots_evaluated', 'validation_slots_evaluated')})
        shutil.copyfile(resume_from / 'best.pt', output / 'best.pt')
        shutil.copyfile(resume_from / 'latest.pt', output / 'latest.pt')

    def status(**values):
        metadata.update(values)
        metadata['elapsed_s'] = elapsed_before + perf_counter() - started
        write_json(output / 'metadata.json', metadata)

    def snapshot(iteration):
        return dict(iteration=iteration, policy=agent.state_dict(), optimizer=agent.optimizer.state_dict(),
                    targets=torch.from_numpy(agent.targets()), torch_rng=torch.get_rng_state(),
                    best_validation_delay_s=best_delay, elapsed_s=elapsed_before + perf_counter() - started,
                    train_slots_evaluated=metadata['train_slots_evaluated'],
                    validation_slots_evaluated=metadata['validation_slots_evaluated'])

    status(iteration=first_iteration - 1)
    print(f'Output directory: {output.resolve()}', flush=True)
    try:
        with ProcessPoolExecutor(config['workers'], initializer=initialize_worker,
                                 initargs=(output / 'frozen_members.pt',)) as pool:
            def validate(iteration):
                nonlocal best_delay
                status(phase='validation', iteration=iteration)
                rows = list(pool.map(evaluate_job, [(agent.targets(), epoch, config['evaluation_slots'], 'learned')
                                                    for epoch in config['validation_epochs']]))
                metadata['validation_slots_evaluated'] += len(rows) * config['evaluation_slots']
                delay = float(np.mean([row['mean_delay_s'] for row in rows]))
                write_json(output / f'validation_{iteration:04d}.json', rows)
                if delay < best_delay:
                    best_delay = delay
                    save_checkpoint(output / 'best.pt', snapshot(iteration))
                return delay

            if resume_from is None:
                validate(0)
                save_checkpoint(output / 'latest.pt', snapshot(0))
            fields = ['iteration', 'train_epoch', 'train_sample_mean_delay_s', 'train_sample_min_delay_s',
                      'validation_delay_s', 'best_validation_delay_s', 'exploration_std', 'kl',
                      'policy_loss', 'accepted_steps', 'reward_std', 'train_slots_evaluated',
                      'validation_slots_evaluated', 'elapsed_s']
            with (output / 'updates.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for iteration in range(first_iteration, config['updates'] + 1):
                    status(phase='rollout', iteration=iteration, best_validation_delay_s=best_delay)
                    std = exploration_std(config, iteration)
                    actions, old_mean = agent.sample(config['pairs'], std)
                    targets = actions.sigmoid().reshape(-1, *simulator.initial.shape).numpy()
                    epoch = config['train_epoch_start'] + (iteration - 1) % config['train_epoch_count']
                    rows = list(pool.map(evaluate_job, [(points, epoch, config['train_slots'], 'learned')
                                                        for points in targets]))
                    metadata['train_slots_evaluated'] += len(rows) * config['train_slots']
                    delays = torch.tensor([row['mean_delay_s'] for row in rows])
                    stats = agent.update(actions, -delays, old_mean, std, epochs=config['ppo_epochs'],
                                         clip_ratio=config['clip_ratio'], target_kl=config['target_kl'])
                    validation = None
                    if iteration == 1 or iteration % config['validation_interval'] == 0:
                        validation = validate(iteration)
                    row = dict(iteration=iteration, train_epoch=epoch,
                               train_sample_mean_delay_s=float(delays.mean()), train_sample_min_delay_s=float(delays.min()),
                               validation_delay_s=validation, best_validation_delay_s=best_delay,
                               exploration_std=std, **stats,
                               train_slots_evaluated=metadata['train_slots_evaluated'],
                               validation_slots_evaluated=metadata['validation_slots_evaluated'],
                               elapsed_s=elapsed_before + perf_counter() - started)
                    writer.writerow(row)
                    stream.flush()
                    save_checkpoint(output / 'latest.pt', snapshot(iteration))
                    status(phase='update_completed', iteration=iteration, best_validation_delay_s=best_delay,
                           latest_sample_delay_s=row['train_sample_mean_delay_s'])
                    print(json.dumps(row), flush=True)
            comparisons = None
            if config['final_test']:
                status(phase='final_test')
                comparisons = final_evaluation(pool, output, config)
                metadata['test_slots_evaluated'] = 4 * len(config['test_epochs']) * config['evaluation_slots']
            status(status='completed', phase='done', comparisons=comparisons,
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
               error=f'{type(error).__name__}: {error}')
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/methods/centralized_ppo.toml')
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--updates', type=int)
    parser.add_argument('--skip-test', action='store_true')
    args = parser.parse_args()
    if args.resume_from is None:
        with args.config.open('rb') as stream:
            config = tomllib.load(stream)['train']
    else:
        config = json.loads((args.resume_from / 'config.json').read_text(encoding='utf-8'))
        for key in ('member_sha256', 'member_episode', 'environment', 'action', 'algorithm', 'resume_from'):
            config.pop(key)
    if args.updates is not None:
        config['updates'] = args.updates
    if args.skip_test:
        config['final_test'] = False
    train(config, output_dir=args.output_dir, resume_from=args.resume_from)


if __name__ == '__main__':
    main()
