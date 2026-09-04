"""Training CLI orchestration, metrics and held-out checkpoint selection."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from env.settings import ChannelConfig, DAGConfig, RegionConfig, SchedulingConfig, UAVConfig, UserConfig
from .adapter import DelayEnvironment
from .ppo import PPOSettings
from .runner import Learner, collect_master, collect_member, evaluate


ROOT = Path(__file__).resolve().parents[2]
METRIC_FIELDS = ['mean_delay_s', 'reward', 'failure_rate', 'max_delay_s', 'ground_fraction',
                 'flight_rejected', 'boundary_violation_rate', 'migrations']
LOG_FIELDS = ['stage', 'update', *METRIC_FIELDS, 'actor_loss', 'value_loss', 'approx_kl',
              'entropy', 'early_stop', 'eval_delay_s', 'elapsed_s']


def environment_config():
    return {name: asdict(kind.default()) for name, kind in [
        ('region', RegionConfig), ('uav', UAVConfig), ('user', UserConfig),
        ('dag', DAGConfig), ('channel', ChannelConfig), ('scheduling', SchedulingConfig)]}


def check_environment(saved):
    if json.dumps(saved, sort_keys=True) != json.dumps(environment_config(), sort_keys=True):
        raise ValueError('Default environment configuration differs from this checkpoint')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def fresh_evaluation(learner, count, offset=0, flight=False, baseline=None):
    return evaluate(DelayEnvironment(offset), learner, count, flight, baseline)


def parser():
    result = argparse.ArgumentParser(description='Additive default-scene Master/Member PPO baseline')
    result.add_argument('--output', type=Path)
    result.add_argument('--resume', type=Path, help='Resume latest.pt; counts are total target updates')
    result.add_argument('--member-updates', type=int, default=100)
    result.add_argument('--master-updates', type=int, default=30)
    result.add_argument('--rollout-slots', type=int, default=8)
    result.add_argument('--eval-slots', type=int, default=4)
    result.add_argument('--test-slots', type=int, default=16)
    result.add_argument('--eval-every', type=int, default=10)
    result.add_argument('--hidden', type=int, default=64)
    result.add_argument('--epochs', type=int, default=4)
    result.add_argument('--seed', type=int, default=7, help='RL seed; environment seeds stay at defaults')
    result.add_argument('--threads', type=int, default=1)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.member_updates, args.master_updates) < 0:
        raise ValueError('Update counts must be nonnegative')
    if any(getattr(args, key) < 1 for key in ('rollout_slots', 'eval_slots', 'test_slots',
                                             'eval_every', 'hidden', 'epochs', 'threads')):
        raise ValueError('Slot counts, intervals, widths, epochs and threads must be positive')
    output = (args.output or (args.resume.parent if args.resume else ROOT / 'scripts/output/rl-baseline')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress = dict(member_updates=0, master_updates=0, master_started=False,
                    best_member_delay=1e30, best_master_delay=1e30)
    if args.resume:
        if output != args.resume.resolve().parent:
            raise ValueError('Resume in the checkpoint directory so its selected Member is available')
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        config = saved['metadata']['config']
        check_environment(config['environment'])
        progress = saved['metadata']['progress'].copy()
        if progress['master_started'] and args.member_updates > progress['member_updates']:
            raise ValueError('Cannot resume Member updates after Master training has frozen it')
        if args.member_updates < progress['member_updates'] or args.master_updates < progress['master_updates']:
            raise ValueError('Requested total update counts precede the checkpoint')
        # Resume uses recorded optimization/data settings; only total update targets change.
        training = config['training']
        for key in ('rollout_slots', 'eval_slots', 'test_slots', 'eval_every', 'hidden', 'epochs', 'seed', 'threads'):
            setattr(args, key, training[key])
        offset = saved['metadata']['workload_slots']
    else:
        if any(output.iterdir()):
            raise FileExistsError('Output directory is not empty; use --resume or a new --output')
        config = dict(environment=environment_config(), training={
            key: getattr(args, key) for key in ('rollout_slots', 'eval_slots', 'test_slots',
                                               'eval_every', 'hidden', 'epochs', 'seed', 'threads')},
            objective='negative mean censored DAG delay / slot duration',
            member_ground_bias_initial=4., torch_version=str(torch.__version__))
        offset = args.eval_slots + args.test_slots
    config['training'].update(member_updates=args.member_updates, master_updates=args.master_updates)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    env = DelayEnvironment(offset)
    learner = Learner(env, args.hidden, PPOSettings(epochs=args.epochs))
    if args.resume:
        learner.load(args.resume)
    write_json(output / 'config.json', config)
    start = time.monotonic()

    def metadata(flight=False):
        return dict(config=config, progress=progress.copy(), workload_slots=env.workload_slots,
                    flight=flight)

    def log(stage, update, metrics, losses=None, validation=None):
        row = dict(stage=stage, update=update, **metrics, **(losses or {}),
                   elapsed_s=time.monotonic() - start)
        if validation is not None:
            row['eval_delay_s'] = validation['mean_delay_s']
        log_path = output / 'metrics.csv'
        has_header = log_path.exists()
        with log_path.open('a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, LOG_FIELDS)
            if not has_header:
                writer.writeheader()
            writer.writerow(row)
        message = (f"{stage} {update:4d} | delay={metrics['mean_delay_s']:.6f}s "
                   f"failed={metrics['failure_rate']:.1%} ground={metrics['ground_fraction']:.1%}")
        if validation is not None:
            message += f" | eval={validation['mean_delay_s']:.6f}s"
        print(message, flush=True)

    if not args.resume:
        initial = fresh_evaluation(learner, args.eval_slots)
        progress['best_member_delay'] = initial['mean_delay_s']
        learner.save(output / 'member_best.pt', metadata())
        log('initial_member', 0, initial, validation=initial)
        baselines = {name: fresh_evaluation(learner, args.eval_slots, baseline=name)
                     for name in ('ground', 'owner')}
        write_json(output / 'validation_baselines.json', baselines)
        print('Validation baselines: ' + json.dumps(baselines), flush=True)

    for update in range(progress['member_updates'] + 1, args.member_updates + 1):
        rollout, rows = collect_member(env, learner, args.rollout_slots)
        losses = learner.member_ppo.update(rollout)
        progress['member_updates'] = update
        validation = None
        if update % args.eval_every == 0 or update == args.member_updates:
            validation = fresh_evaluation(learner, args.eval_slots)
            if validation['mean_delay_s'] < progress['best_member_delay']:
                progress['best_member_delay'] = validation['mean_delay_s']
                learner.save(output / 'member_best.pt', metadata())
        log('member', update, {key: float(np.mean([row[key] for row in rows])) for key in METRIC_FIELDS},
            losses, validation)
        learner.save(output / 'latest.pt', metadata())

    if args.master_updates and not progress['master_started']:
        selected = torch.load(output / 'member_best.pt', map_location='cpu', weights_only=True)
        learner.member.load_state_dict(selected['member'])
        learner.member_value.load_state_dict(selected['member_value'])
        progress['master_started'] = True
        initial = fresh_evaluation(learner, args.eval_slots, flight=True)
        progress['best_master_delay'] = initial['mean_delay_s']
        learner.save(output / 'best.pt', metadata(flight=True))
        log('initial_master', 0, initial, validation=initial)

    for update in range(progress['master_updates'] + 1, args.master_updates + 1):
        rollout, rows = collect_master(env, learner, args.rollout_slots)
        losses = learner.master_ppo.update(rollout)
        progress['master_updates'] = update
        validation = None
        if update % args.eval_every == 0 or update == args.master_updates:
            validation = fresh_evaluation(learner, args.eval_slots, flight=True)
            if validation['mean_delay_s'] < progress['best_master_delay']:
                progress['best_master_delay'] = validation['mean_delay_s']
                learner.save(output / 'best.pt', metadata(flight=True))
        log('master', update, {key: float(np.mean([row[key] for row in rows])) for key in METRIC_FIELDS},
            losses, validation)
        learner.save(output / 'latest.pt', metadata(flight=True))

    # latest is resumable training state; best is validation-selected deployment state.
    learner.save(output / 'latest.pt', metadata(flight=progress['master_started']))
    learner.save(output / 'final.pt', metadata(flight=progress['master_started']))
    if not progress['master_started']:
        learner.load(output / 'member_best.pt', optimizers=False, restore_rng=False)
        learner.save(output / 'best.pt', metadata())
    selected_metadata = learner.load(output / 'best.pt', optimizers=False, restore_rng=False)
    test_results = {name: fresh_evaluation(learner, args.test_slots, offset=args.eval_slots, baseline=name)
                    for name in ('ground', 'owner')}
    test_results['policy'] = fresh_evaluation(learner, args.test_slots, offset=args.eval_slots,
                                             flight=selected_metadata['flight'])
    write_json(output / 'summary.json', dict(completed=progress, environment=config['environment'],
                                              test=test_results, elapsed_s=time.monotonic() - start))
    print('Held-out test: ' + json.dumps(test_results), flush=True)
    print(f'Checkpoints and metrics: {output}', flush=True)


def evaluation_main(argv=None):
    arguments = argparse.ArgumentParser(description='Evaluate RL checkpoint on reserved DAGs')
    arguments.add_argument('--checkpoint', required=True, type=Path)
    arguments.add_argument('--split', choices=('validation', 'test'), default='test')
    arguments.add_argument('--slots', type=int, help='Defaults to the saved size of the selected split')
    arguments.add_argument('--output', type=Path, help='Optional JSON report path')
    args = arguments.parse_args(argv)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = state['metadata']['config']
    check_environment(config['environment'])
    training = config['training']
    maximum = training['test_slots' if args.split == 'test' else 'eval_slots']
    count = args.slots if args.slots is not None else maximum
    if not 1 <= count <= maximum:
        raise ValueError(f'--slots must be in 1..{maximum} to remain in the reserved split')
    offset = training['eval_slots'] if args.split == 'test' else 0
    torch.set_num_threads(training['threads'])
    env = DelayEnvironment(offset)
    learner = Learner(env, state['hidden'], PPOSettings(**state['settings']))
    metadata = learner.load(args.checkpoint, optimizers=False, restore_rng=False)
    metrics = evaluate(env, learner, count, metadata['flight'])
    report = dict(checkpoint=str(args.checkpoint.resolve()), split=args.split, slots=count,
                  flight=metadata['flight'], **metrics)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
