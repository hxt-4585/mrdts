"""分层 PPO：Master 参数共享 CTDE，Member 独立 PPO；训练和确定性评估。

python methods/hmappo/run.py --episodes 1 --slots 2 --users 4 --dag-nodes 3
python methods/hmappo/run.py --evaluate results/hmappo/.../checkpoint.pt --episodes 1
python methods/hmappo/run.py --config config/methods/hmappo_staged.toml
python methods/hmappo/run.py --config config/methods/hmappo_staged.toml --stage master --member-checkpoint PATH

默认参数见 config/methods/hmappo.toml。仅采样当前策略，不跨更新复用旧 rollout。
Actor 只使用局部观测；Master critic 使用全局摘要，Member critic 使用自身局部观测。
检查点包含环境配置快照，评估不读取之后修改的公共配置；epoch_start 默认
从训练 episode 数之后开始，以使用不同 DAG 随机流，空间场景保持一致。
"""

import argparse
import csv
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter
import tomllib
import uuid

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ''):
    sys.path.insert(0, str(PROJECT_ROOT))

from env.settings import ChannelConfig, DAGConfig, EnvConfig, RegionConfig, UAVConfig, UserConfig
from env.settings.uav_config import MemberPropulsionConfig
from methods.hmappo.support import build_scene, resolved_settings, MetricSummary, write_json, git_output
from methods.hmappo.ppo import finish_trajectory, update
from methods.hmappo.rollout import Adapter, collect_slot, master_bootstrap
from methods.hmappo.curriculum import FlightCurriculum, scenario_settings


def output_directory(root, mode, seed):
    root = Path(root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
    output = root / 'hmappo' / f'seed_{seed}' / f'{mode}_{stamp}'
    output.mkdir(parents=True)
    print(f'Output directory: {output}', flush=True)
    return output


def settings_from_snapshot(snapshot):
    settings = {name: cls(**snapshot[name]) for name, cls in (
        ('region', RegionConfig), ('user', UserConfig), ('channel', ChannelConfig),
        ('dag', DAGConfig), ('env', EnvConfig))}
    uav = dict(snapshot['uav'])
    uav['member_propulsion'] = MemberPropulsionConfig(**uav['member_propulsion'])
    settings['uav'] = UAVConfig(**uav)
    return settings


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    env = build_scene(settings_from_snapshot(checkpoint['environment']), episode=0)
    master, members = Adapter(env).policies(checkpoint['config']['hidden'], device)
    master.load_state_dict(checkpoint['master'])
    members.load_state_dict(checkpoint['member_policies'])
    master.eval()
    members.eval()
    return env, master, members, checkpoint


def train(*, episodes=20, slots=None, users=None, dag_nodes=None, seed=45, device='cpu',
          hidden=128, rollout_slots=8, ppo_epochs=4, minibatch_size=256,
          learning_rate=3e-4, master_gamma=.99, member_gamma=1., gae_lambda=.95,
          clip_ratio=.2, entropy_coef=.01, value_coef=.5, max_grad_norm=.5,
          torch_threads=1, output_root='results', boundary_penalty=.1,
          diagnostics_interval=5, log_interval=5, stage='joint', member_checkpoint=None,
          scene_seeds=(42, 43, 44, 45)):
    config = dict(episodes=episodes, slots=slots, users=users, dag_nodes=dag_nodes, seed=seed,
                  device=device, hidden=hidden, rollout_slots=rollout_slots, ppo_epochs=ppo_epochs,
                  minibatch_size=minibatch_size, learning_rate=learning_rate, master_gamma=master_gamma,
                  member_gamma=member_gamma, gae_lambda=gae_lambda, clip_ratio=clip_ratio,
                  entropy_coef=entropy_coef, value_coef=value_coef, max_grad_norm=max_grad_norm,
                  torch_threads=torch_threads, output_root=str(output_root), boundary_penalty=boundary_penalty,
                  diagnostics_interval=diagnostics_interval, log_interval=log_interval,
                  stage=stage, member_checkpoint=str(member_checkpoint) if member_checkpoint else None,
                  scene_seeds=list(scene_seeds))
    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    settings = resolved_settings(users, dag_nodes)
    if stage == 'master':
        source = torch.load(member_checkpoint, map_location=device, weights_only=True)
        settings = settings_from_snapshot(source['environment'])
        hidden = source['config']['hidden']
        config['hidden'] = hidden
    if slots is not None:
        settings['env'] = replace(settings['env'], steps_per_epoch=slots)
    env = build_scene(settings, episode=0)
    adapter = Adapter(env)
    master, members = adapter.policies(hidden, device)
    if stage == 'master':
        members.load_state_dict(source['member_policies'])
        members.requires_grad_(False)
        members.eval()
    if stage == 'member':
        master.requires_grad_(False)
        master.eval()
    master_optimizer = (torch.optim.Adam(master.parameters(), lr=learning_rate, eps=1e-5)
                        if stage != 'member' else None)
    member_optimizers = [torch.optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)
                         for policy in members] if stage != 'master' else []
    output = output_directory(output_root, 'train' if stage == 'joint' else 'train_' + stage, seed)
    print(f'HMAPPO | Environment seed {settings["env"].env_seed} | Method seed {seed} | '
          f'{episodes} episodes | {env.env_config.steps_per_epoch} slots/episode', flush=True)
    snapshot = {name: asdict(value) for name, value in settings.items()}
    write_json(output / 'config.json', dict(method='hmappo', training=config, environment=snapshot,
        ordering='ERS', scenario_distribution='curriculum' if stage == 'member' else 'fixed',
        frozen_members=stage == 'master',
        master_reward='environment.master_reward - boundary_penalty * preflight_owned_violations',
        member_reward='environment.member_rewards', member_terminal='end of scheduling slot',
        discount_unit='one decision per agent', architecture='master CTDE; independent member PPO by UAV ID',
        policy_inputs='local observation excluding energy queues; master appends ownership mask and identity',
        critic_inputs='master: global positions, regions, epoch progress and identity; '
                      'member: same local observation as its actor',
        logging=dict(schema_version=2, empty='undefined, no samples, or detailed diagnostics not collected',
                     indexing='episode and slot in metrics: zero-based; update: one-based; update slot: exclusive end',
                     kl='legacy minibatch pre-step average; final_kl: full rollout after all optimizer steps',
                     gradients='pre-clipping norms; actor includes continuous log_std; scale uses clip_grad_norm epsilon',
                     moments='population std; advantage before normalization, active samples only',
                     actions='continuous: tanh actions on owned dimensions; discrete: rollout selection fractions',
                     value='explained variance against GAE targets, before/after update; constant targets undefined',
                     detailed='master every update; members at diagnostics_interval and episode end',
                     flush='metrics and updates at each completed update; episodes at episode end')))
    metadata = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                    branch=git_output('branch', '--show-current'), commit=git_output('rev-parse', 'HEAD'),
                    torch=str(torch.__version__), device=device)
    write_json(output / 'metadata.json', metadata)
    summary, updates, last_update = MetricSummary(), 0, {}
    update_options = dict(ppo_epochs=ppo_epochs, minibatch_size=minibatch_size, clip_ratio=clip_ratio,
                          entropy_coef=entropy_coef, value_coef=value_coef, max_grad_norm=max_grad_norm)
    started = perf_counter()
    try:
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as metric_file, \
                (output / 'updates.csv').open('w', newline='', encoding='utf-8') as update_file, \
                (output / 'episodes.csv').open('w', newline='', encoding='utf-8') as episode_file:
            metric_writer = update_writer = episode_writer = None
            for episode in range(episodes):
                flight = None
                if stage == 'member':
                    episode_settings = scenario_settings(settings, episode, scene_seeds)
                    env = build_scene(episode_settings, episode=episode)
                    adapter = Adapter(env)
                env.reset(epoch_index=episode)
                if stage == 'member':
                    flight = FlightCurriculum(env, episode, seed)
                    with (output / 'scenarios.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(dict(episode=episode, flight_mode=flight.mode,
                            environment={name: asdict(value) for name, value in episode_settings.items()})) + '\n')
                episode_summary = MetricSummary()
                for start in range(0, env.env_config.steps_per_epoch, rollout_slots):
                    rollout_started = perf_counter()
                    master_trajectories = [[] for _ in range(adapter.masters)]
                    member_samples = [[] for _ in members]
                    stop = min(start + rollout_slots, env.env_config.steps_per_epoch)
                    for _ in range(start, stop):
                        high, low, row = collect_slot(adapter, master, members,
                                                     member_gamma=member_gamma, gae_lambda=gae_lambda,
                                                     boundary_penalty=boundary_penalty,
                                                     flight_actions=flight.actions(env) if flight else None,
                                                     member_deterministic=stage == 'master')
                        if stage == 'member':
                            row.update(scene_seed=env.env_config.env_seed, flight_mode=flight.mode)
                        for trajectory, sample in zip(master_trajectories, high):
                            trajectory.append(sample)
                        for owner, samples in low.items():
                            member_samples[owner].extend(samples)
                        summary.add(row)
                        episode_summary.add(row)
                        if metric_writer is None:
                            metric_writer = csv.DictWriter(metric_file, fieldnames=list(row))
                            metric_writer.writeheader()
                        metric_writer.writerow(row)
                    bootstrap = (np.zeros(adapter.masters) if stage == 'member' or stop == env.env_config.steps_per_epoch
                                 else master_bootstrap(adapter, master))
                    master_samples = [sample for trajectory, value in zip(master_trajectories, bootstrap)
                                      for sample in finish_trajectory(trajectory, value, master_gamma, gae_lambda)]
                    rollout_elapsed = perf_counter() - rollout_started
                    update_started = perf_counter()
                    # 本轮采样全部结束后再更新两层，保持同一 rollout 的行为策略固定。
                    stats = update(master, master_optimizer, master_samples, **update_options)
                    last_update = {f'master_{key}': value for key, value in stats.items()}
                    for owner, (trajectory, value) in enumerate(zip(master_trajectories, bootstrap)):
                        active_samples = [s for s in trajectory if s.mask.any()]
                        last_update.update({f'master_{owner}_active_samples': len(active_samples),
                                            f'master_{owner}_bootstrap': float(value),
                                            f'master_{owner}_reward_mean': float(np.mean([s.reward for s in trajectory])) if trajectory else None,
                                            f'master_{owner}_advantage_mean': (float(np.mean([s.advantage for s in active_samples]))
                                                                               if active_samples else None),
                                            f'master_{owner}_advantage_std': (float(np.std([s.advantage for s in active_samples]))
                                                                              if active_samples else None)})
                    detailed = (updates + 1) % diagnostics_interval == 0 or stop == env.env_config.steps_per_epoch
                    for owner, (policy, samples) in enumerate(zip(members, member_samples)):
                        optimizer = member_optimizers[owner] if stage != 'master' else None
                        if stage == 'master':
                            samples = []
                        stats = update(policy, optimizer, samples, diagnostics=detailed, **update_options)
                        last_update.update({f'member_{owner}_{key}': value for key, value in stats.items()})
                    updates += 1
                    update_elapsed = perf_counter() - update_started
                    update_row = dict(update=updates, episode=episode, slot=stop,
                                      rollout_elapsed_s=rollout_elapsed, update_elapsed_s=update_elapsed,
                                      **last_update)
                    if update_writer is None:
                        update_writer = csv.DictWriter(update_file, fieldnames=list(update_row))
                        update_writer.writeheader()
                    update_writer.writerow(update_row)
                    metric_file.flush()
                    update_file.flush()
                    if updates % log_interval == 0 or stop == env.env_config.steps_per_epoch:
                        master_ev = last_update['master_explained_variance_after']
                        ev_text = 'n/a' if master_ev is None else f'{master_ev:.3f}'
                        member_counts = [len(samples) for samples in member_samples]
                        member_kl = [last_update[f'member_{i}_final_kl'] for i in range(len(members))
                                     if last_update[f'member_{i}_final_kl'] is not None]
                        low_kl = f'{max(member_kl):.4f}' if member_kl else 'n/a'
                        member_ev = [last_update[f'member_{i}_explained_variance_after'] for i in range(len(members))
                                     if last_update[f'member_{i}_explained_variance_after'] is not None]
                        low_ev = f'{min(member_ev):.3f}' if member_ev else 'n/a'
                        high_reward = last_update['master_reward_mean']
                        high_kl = last_update['master_final_kl']
                        high_clip = last_update['master_final_clip_fraction']
                        high_text = (f'R {high_reward:.3f} KL {high_kl:.4f} Clip {high_clip:.3f}'
                                     if high_reward is not None else 'frozen/external')
                        print(f'Update {updates} | Episode {episode + 1}/{episodes} Slot {stop} | '
                              f'Delay {episode_summary.result()["mean_delay_s"]:.3f}s | '
                              f'Master {high_text} EV {ev_text} | '
                              f'Member samples {min(member_counts)}..{max(member_counts)} '
                              f'inactive {member_counts.count(0)} KLmax {low_kl} EVmin {low_ev} | '
                              f'Rollout {rollout_elapsed:.2f}s Update {update_elapsed:.2f}s', flush=True)
                episode_row = dict(episode=episode, update=updates, **episode_summary.result())
                if episode_writer is None:
                    episode_writer = csv.DictWriter(episode_file, fieldnames=list(episode_row))
                    episode_writer.writeheader()
                episode_writer.writerow(episode_row)
                episode_file.flush()
                print(f'Episode {episode + 1} summary | Delay {episode_row["mean_delay_s"]:.3f}s | '
                      f'Failure {episode_row["failure_rate"]:.3f} | Violations {episode_row["flight_violations"]} | '
                      f'Master R {episode_row["master_training_reward_mean"]:.3f}', flush=True)
                # 每个完整 episode 保存；临时文件替换避免中断损坏上一份检查点。
                checkpoint = dict(config=config, environment=snapshot, episodes_completed=episode + 1,
                    updates=updates, last_update=last_update, master=master.state_dict(),
                    member_policies=members.state_dict(), master_optimizer=master_optimizer.state_dict() if master_optimizer else None,
                    member_optimizers=[optimizer.state_dict() for optimizer in member_optimizers])
                torch.save(checkpoint, output / 'checkpoint.tmp')
                (output / 'checkpoint.tmp').replace(output / 'checkpoint.pt')
        write_json(output / 'summary.json', dict(**summary.result(), updates=updates,
                                               elapsed_s=perf_counter() - started))
        metadata['status'] = 'completed'
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        metadata.update(finished_at=datetime.now(timezone.utc).isoformat(),
                        slots_completed=summary.slots, updates=updates)
        write_json(output / 'metadata.json', metadata)
    return output


def evaluate(checkpoint_path, *, episodes=2, device='cpu', output_root='results', epoch_start=None,
             torch_threads=1, scene_seeds=None):
    torch.set_num_threads(torch_threads)
    env, master, members, checkpoint = load_checkpoint(checkpoint_path, device)
    adapter = Adapter(env)
    member_stage = checkpoint['config'].get('stage') == 'member'
    if member_stage and scene_seeds is None:
        scene_seeds = [value + 10000 for value in checkpoint['config']['scene_seeds']]
    if epoch_start is None:
        epoch_start = checkpoint['episodes_completed']
    output = output_directory(output_root, 'eval', checkpoint['config']['seed'])
    write_json(output / 'config.json', dict(checkpoint=str(Path(checkpoint_path).resolve()),
        episodes=episodes, epoch_start=epoch_start, deterministic=True, environment=checkpoint['environment'],
        boundary_penalty=checkpoint['config']['boundary_penalty'], scene_seeds=scene_seeds,
        flight_policy='curriculum' if member_stage else 'learned'))
    summary = MetricSummary()
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = None
        for episode in range(epoch_start, epoch_start + episodes):
            flight = None
            if member_stage:
                settings = scenario_settings(settings_from_snapshot(checkpoint['environment']),
                                             episode - epoch_start, scene_seeds)
                env = build_scene(settings, episode=episode)
                adapter = Adapter(env)
            env.reset(epoch_index=episode)
            if member_stage:
                flight = FlightCurriculum(env, episode - epoch_start, checkpoint['config']['seed'] + 10000)
            for _ in range(env.env_config.steps_per_epoch):
                _, _, row = collect_slot(adapter, master, members, deterministic=True,
                                         boundary_penalty=checkpoint['config']['boundary_penalty'],
                                         flight_actions=flight.actions(env) if flight else None)
                if member_stage:
                    row.update(scene_seed=env.env_config.env_seed, flight_mode=flight.mode, epoch=episode)
                summary.add(row)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if env.step_index % 8 == 0 or env.step_index == env.env_config.steps_per_epoch:
                    stream.flush()
                    print(f'Eval episode {episode} | Slot {env.step_index} | '
                          f'Mean delay {row["mean_delay_s"]:.3f}s', flush=True)
    write_json(output / 'summary.json', summary.result())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'config' / 'methods' / 'hmappo.toml')
    parser.add_argument('--evaluate', type=Path, help='加载检查点，执行确定性评估')
    parser.add_argument('--epoch-start', type=int, help='评估 DAG 随机流的起始 episode')
    parser.add_argument('--stage', choices=('joint', 'member', 'master'), help='联合训练/低层预训练/冻结低层训练高层')
    parser.add_argument('--member-checkpoint', type=Path, help='master 阶段加载的低层检查点')
    parser.add_argument('--scene-seeds', type=int, nargs='+', help='低层场景种子；评估时作为留出种子')
    for option in ('episodes', 'slots', 'users', 'dag-nodes', 'seed', 'hidden', 'rollout-slots',
                   'ppo-epochs', 'minibatch-size', 'torch-threads', 'diagnostics-interval', 'log-interval'):
        parser.add_argument('--' + option, type=int)
    for option in ('learning-rate', 'master-gamma', 'member-gamma', 'gae-lambda', 'clip-ratio',
                   'entropy-coef', 'value-coef', 'max-grad-norm', 'boundary-penalty'):
        parser.add_argument('--' + option, type=float)
    parser.add_argument('--device')
    parser.add_argument('--output-root')
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    with config_path.open('rb') as stream:
        config = tomllib.load(stream)['train']
    config.update({key: value for key, value in vars(args).items()
                   if key not in ('config', 'evaluate', 'epoch_start') and value is not None})
    if args.evaluate is not None:
        output = evaluate(args.evaluate, episodes=args.episodes if args.episodes is not None else 2,
                          device=config['device'], output_root=config['output_root'],
                          epoch_start=args.epoch_start, torch_threads=config['torch_threads'],
                          scene_seeds=args.scene_seeds)
    else:
        output = train(**config)
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    main()
