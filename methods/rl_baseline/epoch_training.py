"""CLI for epochs of 500 complete environment steps; optimizer epochs are separate."""

import argparse
import csv
from pathlib import Path
import sys
import time

import torch

from .adapter import DelayEnvironment
from .epoch_runner import EpochLearner, run_epoch
from .ppo import PPOSettings
from .training import ROOT, METRIC_FIELDS, check_environment, environment_config, write_json


EPOCH_FIELDS = ['epoch', 'stage', 'stage_epoch', 'steps', 'total_steps', 'master_calls',
                'scheduled_tasks', 'epoch_return', 'mean_reward', *METRIC_FIELDS,
                'member_updates', 'master_updates', 'actor_loss', 'value_loss',
                'approx_kl', 'entropy', 'early_stop', 'validation_mean_reward', 'elapsed_s']
STEP_FIELDS = ['epoch', 'stage', 'stage_epoch', 'global_step', 'step', 'slot_start', 'slot_end',
               'master_calls', 'scheduled_dags', 'scheduled_tasks', *METRIC_FIELDS]
SETTINGS = ('steps_per_epoch', 'update_every_steps', 'eval_steps', 'test_steps', 'eval_every',
            'hidden', 'ppo_epochs', 'seed', 'threads', 'log_every_steps')


def parser():
    result = argparse.ArgumentParser(description='500 full Master-command + DAG-scheduling steps per epoch')
    result.add_argument('--output', type=Path)
    result.add_argument('--resume', type=Path)
    result.add_argument('--member-epochs', type=int, default=100)
    result.add_argument('--master-epochs', type=int, default=30)
    result.add_argument('--steps-per-epoch', type=int, default=500)
    result.add_argument('--update-every-steps', type=int, default=8,
                        help='Member buffer flush interval; does not reset the epoch or positions')
    result.add_argument('--ppo-epochs', type=int, default=4, help='Optimizer passes over each collected batch')
    result.add_argument('--eval-steps', type=int, default=500)
    result.add_argument('--test-steps', type=int, default=500)
    result.add_argument('--eval-every', type=int, default=10)
    result.add_argument('--hidden', type=int, default=64)
    result.add_argument('--seed', type=int, default=7)
    result.add_argument('--threads', type=int, default=1)
    result.add_argument('--log-every-steps', type=int, default=25)
    return result


def append_csv(path, fields, row):
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def validate_checkpoint(state):
    metadata = state['metadata']
    if metadata.get('kind') != 'full_step_epochs_v1':
        raise ValueError('Use a checkpoint created by train_rl_epochs.py')
    check_environment(metadata['config']['environment'])
    return metadata


def reconcile_logs(output, completed_epoch):
    """Keep logs aligned with the last completed checkpoint after interruption."""
    for filename in ('steps.csv', 'epochs.csv'):
        path = output / filename
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames
            rows = list(reader)
        kept = [row for row in rows if int(row['epoch']) <= completed_epoch]
        discarded = [row for row in rows if int(row['epoch']) > completed_epoch]
        if not discarded:
            continue
        # Preserve unfinished-attempt evidence, but do not double-count it as training.
        backup = output / f'{path.stem}.uncheckpointed-{time.time_ns()}.csv'
        with backup.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fields)
            writer.writeheader()
            writer.writerows(discarded)
        temporary = path.with_suffix('.resume.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fields)
            writer.writeheader()
            writer.writerows(kept)
        temporary.replace(path)


def evaluate_fresh(learner, steps, offset=0):
    return run_epoch(DelayEnvironment(offset), learner, stage='evaluation', steps=steps)


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if min(args.member_epochs, args.master_epochs) < 0 or any(
            getattr(args, name) < 1 for name in SETTINGS if name != 'seed'):
        arguments.error('Counts and widths must be positive; stage epoch counts may be zero')
    output = (args.output or (args.resume.parent if args.resume else ROOT / 'scripts/output/rl-epochs')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress = dict(epoch=0, total_steps=0, member_epochs=0, master_epochs=0,
                    master_started=False, member_updates=0, master_updates=0,
                    best_member_reward=-1e30, best_master_reward=-1e30)
    if args.resume:
        if output != args.resume.resolve().parent:
            arguments.error('Resume in the existing checkpoint directory')
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        metadata = validate_checkpoint(state)
        config = metadata['config']
        progress = metadata['progress'].copy()
        if progress['master_started'] and args.member_epochs > progress['member_epochs']:
            arguments.error('Member is already frozen for Master training')
        if args.member_epochs < progress['member_epochs'] or args.master_epochs < progress['master_epochs']:
            arguments.error('Target epoch counts cannot precede the saved progress')
        # An episode-length change would also alter the reserved data partitions.
        explicit_steps = any(token == '--steps-per-epoch' or token.startswith('--steps-per-epoch=')
                             for token in (argv if argv is not None else sys.argv[1:]))
        if explicit_steps and args.steps_per_epoch != config['training']['steps_per_epoch']:
            arguments.error('steps-per-epoch cannot change on resume; start a new experiment')
        for name in SETTINGS:
            setattr(args, name, config['training'][name])
        offset = metadata['workload_slots']
    else:
        if any(output.iterdir()):
            raise FileExistsError('Use a new output directory or --resume')
        config = dict(environment=environment_config(), training={name: getattr(args, name) for name in SETTINGS},
                      objective='negative global mean censored DAG delay / slot duration',
                      epoch_definition='500 by default: every step calls Master then Member DAG scheduling',
                      member_ground_bias_initial=4., torch_version=str(torch.__version__))
        offset = args.eval_steps + args.test_steps
    config['training'].update(member_epochs=args.member_epochs, master_epochs=args.master_epochs)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    env = DelayEnvironment(offset)
    learner = EpochLearner(env, args.hidden, PPOSettings(epochs=args.ppo_epochs))
    if args.resume:
        learner.load(args.resume)
        reconcile_logs(output, progress['epoch'])
    write_json(output / 'config.json', config)
    started = time.monotonic()

    def saved_metadata(stage):
        return dict(kind='full_step_epochs_v1', config=config, progress=progress.copy(),
                    workload_slots=env.workload_slots, stage=stage)

    if not args.resume:
        validation = evaluate_fresh(learner, args.eval_steps)
        progress['best_member_reward'] = validation['mean_reward']
        learner.save(output / 'member_best.pt', saved_metadata('member'))
        learner.save(output / 'latest.pt', saved_metadata('member'))
        write_json(output / 'initial_validation.json', validation)
        print(f"Initial validation mean reward: {validation['mean_reward']:.6f}", flush=True)

    for stage, target in [('member', args.member_epochs), ('master', args.master_epochs)]:
        if stage == 'master' and target and not progress['master_started']:
            learner.save(output / 'member_final.pt', saved_metadata('member'))
            selected = torch.load(output / 'member_best.pt', map_location='cpu', weights_only=True)
            learner.member.load_state_dict(selected['member'])
            learner.member_value.load_state_dict(selected['member_value'])
            progress['master_started'] = True
            validation = evaluate_fresh(learner, args.eval_steps)
            progress['best_master_reward'] = validation['mean_reward']
            learner.save(output / 'best.pt', saved_metadata('master'))
            learner.save(output / 'latest.pt', saved_metadata('master'))
        for stage_epoch in range(progress[f'{stage}_epochs'] + 1, target + 1):
            epoch = progress['epoch'] + 1
            base_step = progress['total_steps']

            def record_step(row):
                append_csv(output / 'steps.csv', STEP_FIELDS,
                           dict(epoch=epoch, stage=stage, stage_epoch=stage_epoch,
                                global_step=base_step + row['step'], **row))
                if row['step'] % args.log_every_steps == 0 or row['step'] == args.steps_per_epoch:
                    print(f"epoch {epoch} ({stage}) step {row['step']}/{args.steps_per_epoch} "
                          f"reward={row['reward']:.6f}", flush=True)

            result = run_epoch(env, learner, stage=stage, steps=args.steps_per_epoch,
                                update_every=args.update_every_steps, on_step=record_step)
            progress['epoch'] = epoch
            progress['total_steps'] += result['steps']
            progress[f'{stage}_epochs'] = stage_epoch
            for role in ('member', 'master'):
                progress[f'{role}_updates'] += result[f'{role}_updates']
            validation = None
            if stage_epoch % args.eval_every == 0 or stage_epoch == target:
                validation = evaluate_fresh(learner, args.eval_steps)
                if validation['mean_reward'] > progress[f'best_{stage}_reward']:
                    progress[f'best_{stage}_reward'] = validation['mean_reward']
                    learner.save(output / ('member_best.pt' if stage == 'member' else 'best.pt'), saved_metadata(stage))
            row = dict(epoch=epoch, stage=stage, stage_epoch=stage_epoch,
                       total_steps=progress['total_steps'], elapsed_s=time.monotonic() - started, **result)
            if validation is not None:
                row['validation_mean_reward'] = validation['mean_reward']
            append_csv(output / 'epochs.csv', EPOCH_FIELDS, row)
            learner.save(output / 'latest.pt', saved_metadata(stage))
            print(f"epoch {epoch} complete: steps={result['steps']} mean_reward={result['mean_reward']:.6f} "
                  f"epoch_return={result['epoch_return']:.6f}", flush=True)
    learner.save(output / 'latest.pt', saved_metadata('master' if progress['master_started'] else 'member'))
    learner.save(output / 'final.pt', saved_metadata('master' if progress['master_started'] else 'member'))
    if not progress['master_started']:
        learner.load(output / 'member_best.pt', optimizers=False, restore_rng=False)
        learner.save(output / 'best.pt', saved_metadata('member'))
    learner.load(output / 'best.pt', optimizers=False, restore_rng=False)
    test = evaluate_fresh(learner, args.test_steps, offset=args.eval_steps)
    write_json(output / 'summary.json', dict(progress=progress, test=test,
                                             elapsed_s=time.monotonic() - started))
    print(f"Done: {output}; test mean reward={test['mean_reward']:.6f}", flush=True)


def evaluation_main(argv=None):
    arguments = argparse.ArgumentParser(description='Evaluate a full-step epoch checkpoint')
    arguments.add_argument('--checkpoint', type=Path, required=True)
    arguments.add_argument('--output', type=Path)
    args = arguments.parse_args(argv)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    metadata = validate_checkpoint(state)
    training = metadata['config']['training']
    torch.set_num_threads(training['threads'])
    env = DelayEnvironment(training['eval_steps'])
    learner = EpochLearner(env, state['hidden'], PPOSettings(**state['settings']))
    learner.load(args.checkpoint, optimizers=False, restore_rng=False)
    result = run_epoch(env, learner, stage='evaluation', steps=training['test_steps'])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    import json
    print(json.dumps(result, indent=2))
