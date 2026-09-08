"""Staged training using experiment-owned scenes, runs and held-out episode IDs."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from experiments.artifacts import create_run, write_json
from experiments.metrics import summarize
from experiments.randomness import RandomStreams
from experiments.scene import build_scene, resolved_settings, settings_snapshot
from .checkpoint import SOURCE_COMMIT, read_checkpoint
from .logs import append_csv, metric_rows, reconcile_logs
from .method import PlanningEnvironment
from .rollout import Learner, run_epoch
from .settings import TrainingSettings, training_signature


def fresh_scene(config, episode_id):
    return build_scene(resolved_settings(config), randomness=RandomStreams.from_config(config, episode_id))


def evaluate_fresh(config, learner, steps, episode_id):
    return run_epoch(fresh_scene(config, episode_id), learner, stage='evaluation', steps=steps)


class PPOTrainer:
    def train(self, config, device):
        settings = TrainingSettings.from_config(config)
        from methods.factory import validate_method
        validate_method(config.method)
        if config.checkpoint is not None:
            raise ValueError('Use --resume latest.pt for training; --checkpoint is for evaluation')
        signature = training_signature(config, settings)
        state = None
        if config.resume:
            resume = Path(config.resume).resolve()
            if resume.name != 'latest.pt':
                raise ValueError('Resume only checkpoints/latest.pt; best.pt is for inference')
            state = read_checkpoint(resume)
            saved = state['metadata']
            progress = saved['progress'].copy()
            if progress['master_started'] and settings.member_epochs > progress['member_epochs']:
                raise ValueError('Member is already frozen for Master training')
            if saved['signature'] != signature:
                raise ValueError('Resume environment, split, slot length or learning settings differ from checkpoint')
            if settings.member_epochs < progress['member_epochs'] or settings.master_epochs < progress['master_epochs']:
                raise ValueError('Target epoch counts cannot precede saved progress')
            output = resume.parent.parent
            metadata = json.loads((output/'metadata.json').read_text(encoding='utf-8'))
            metadata.update(status='running', resumed_at=datetime.now(timezone.utc).isoformat())
        else:
            progress = dict(epoch=0, total_steps=0, member_epochs=0, master_epochs=0,
                            member_updates=0, master_updates=0, master_started=False,
                            best_member_reward=-1e30, best_master_reward=-1e30)
            output, metadata = create_run(config, 'train', [settings_snapshot(resolved_settings(config))])
        metadata.update(source_baseline_commit=SOURCE_COMMIT, torch_version=str(torch.__version__),
                        device=str(device), splits=signature['splits'])
        write_json(output/'metadata.json', metadata)
        checkpoints = output/'checkpoints'
        checkpoints.mkdir(exist_ok=True)
        started = time.monotonic()
        try:
            torch.set_num_threads(settings.threads)
            # Policy randomness belongs to the experiment seed. Scene streams remain independent.
            torch.manual_seed(config.seed)
            learner = Learner(PlanningEnvironment(fresh_scene(config, 2)), settings.hidden, settings.ppo(), device)
            learner.selected_checkpoints = {}
            if state is not None:
                learner.load(config.resume)
                reconcile_logs(output, progress['epoch'], master_started=progress['master_started'])
                # latest.pt is the commit point for model selection as well as optimizer state.
                for name, selected in learner.selected_checkpoints.items():
                    if name not in ('member_best.pt', 'best.pt'):
                        raise ValueError('Invalid selected checkpoint name')
                    temporary = (checkpoints/name).with_suffix('.restore.tmp')
                    torch.save(selected, temporary)
                    temporary.replace(checkpoints/name)
            write_json(output/'config.json', dict(experiment=asdict(config),
                       scenes=[settings_snapshot(resolved_settings(config))],
                       resolved_training=asdict(settings), data_splits=signature['splits'],
                       objective='negative global mean censored DAG delay / slot duration',
                       source_baseline_commit=SOURCE_COMMIT))

            def checkpoint_metadata(stage):
                return dict(signature=signature, progress=progress.copy(), stage=stage,
                            training=asdict(settings), run_directory=str(output.resolve()))

            def save(name, stage):
                learner.save(checkpoints/name, checkpoint_metadata(stage))
                if name in ('member_best.pt','best.pt'):
                    selected = read_checkpoint(checkpoints/name)
                    selected.pop('selected_checkpoints', None)
                    learner.selected_checkpoints[name] = selected

            def validation(epoch, stage):
                result = evaluate_fresh(config, learner, settings.eval_steps, 0)
                append_csv(output/'training/validation.csv', dict(epoch=epoch,stage=stage,
                           steps=result['steps'],epoch_return=result['epoch_return'],
                           mean_reward=result['mean_reward'],mean_delay_s=result['mean_delay_s']))
                return result

            if state is None:
                initial = validation(0, 'member')
                progress['best_member_reward'] = initial['mean_reward']
                write_json(output/'training/initial_validation.json', initial)
                save('member_best.pt','member')
                save('latest.pt','member')
                print(f"Initial validation: reward={initial['mean_reward']:.6f}", flush=True)

            for stage, target in (('member',settings.member_epochs),('master',settings.master_epochs)):
                if stage == 'master' and target and not progress['master_started']:
                    save('member_final.pt','member')
                    selected = read_checkpoint(checkpoints/'member_best.pt')
                    learner.member.load_state_dict(selected['member'])
                    learner.member_value.load_state_dict(selected['member_value'])
                    progress['master_started'] = True
                    initial = validation(progress['epoch'],'master')
                    progress['best_master_reward'] = initial['mean_reward']
                    save('best.pt','master')
                    save('latest.pt','master')
                for stage_epoch in range(progress[f'{stage}_epochs']+1, target+1):
                    epoch = progress['epoch']+1
                    episode_id = 2 + epoch - 1
                    base_step = progress['total_steps']

                    def record_step(row):
                        append_csv(output/'metrics.csv', dict(row, episode=epoch-1, epoch=epoch,
                                   stage=stage, stage_epoch=stage_epoch, episode_id=episode_id,
                                   global_step=base_step+row['step']))
                        if row['step'] % settings.log_every_steps == 0 or row['step'] == config.slots:
                            print(f"epoch {epoch} ({stage}) step {row['step']}/{config.slots} reward={row['reward']:.6f}",flush=True)

                    def record_update(row):
                        progress[f'{stage}_updates'] += 1
                        append_csv(output/'training/updates.csv', dict(epoch=epoch, stage=stage,
                                   update=progress['member_updates']+progress['master_updates'],
                                   total_steps=base_step+row['step'],
                                   **{key:value for key,value in row.items() if key != 'step'}))

                    result = run_epoch(fresh_scene(config,episode_id), learner, stage=stage, steps=config.slots,
                                       update_every=settings.update_every_steps,
                                       on_step=record_step, on_update=record_update)
                    progress.update(epoch=epoch,total_steps=base_step+config.slots)
                    progress[f'{stage}_epochs'] = stage_epoch
                    if stage_epoch % settings.eval_every == 0 or stage_epoch == target:
                        report = validation(epoch,stage)
                        if report['mean_reward'] > progress[f'best_{stage}_reward']:
                            progress[f'best_{stage}_reward'] = report['mean_reward']
                            save('member_best.pt' if stage == 'member' else 'best.pt',stage)
                    append_csv(output/'training/epochs.csv', dict(epoch=epoch,stage=stage,stage_epoch=stage_epoch,
                               total_steps=progress['total_steps'],**result))
                    save('latest.pt',stage)
                    write_json(output/'training/progress.json',progress)
                    print(f"epoch {epoch} complete: return={result['epoch_return']:.6f}, mean={result['mean_reward']:.6f}",flush=True)

            stage = 'master' if progress['master_started'] else 'member'
            save('latest.pt',stage)
            save('final.pt',stage)
            if not progress['master_started']:
                learner.load(checkpoints/'member_best.pt',optimizers=False,restore_rng=False)
                save('best.pt','member')
            learner.load(checkpoints/'best.pt',optimizers=False,restore_rng=False)
            test = evaluate_fresh(config,learner,settings.test_steps,1)
            write_json(output/'training/test.json',test)
            write_json(output/'training/progress.json',progress)
            write_json(output/'summary.json',summarize(list(metric_rows(output))))
            metadata.update(status='completed',elapsed_s=time.monotonic()-started)
            print(f"Training complete: {output}\nHeld-out reward={test['mean_reward']:.6f}",flush=True)
        except BaseException as exc:
            metadata.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            metadata['finished_at'] = datetime.now(timezone.utc).isoformat()
            write_json(output/'metadata.json',metadata)
        return output
