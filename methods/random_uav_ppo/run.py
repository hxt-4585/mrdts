"""随机向量飞行 UAV 的独立 Member PPO 基线；无 Master 网络或联合训练入口。

python methods/random_uav_ppo/run.py --config config/methods/random_uav_ppo.toml
python methods/random_uav_ppo/evaluate.py --checkpoints RUN/initial.pt RUN/checkpoint.pt
python methods/random_uav_ppo/plot_member.py RUN
用户布局固定、UAV 定期采样随机二维动作，每轮轨迹不同，训练 DAG 按 episode/slot 变化；评估使用固定留出 DAG 流。
"""

import argparse
import csv
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

from methods.random_uav_ppo.support import settings_from_snapshot
from methods.random_uav_ppo.support import MetricSummary
from methods.random_uav_ppo.support import build_scene, resolved_settings, write_json, git_output
from methods.random_uav_ppo.flight import derive_flight_seed
from methods.random_uav_ppo.ppo import update
from methods.random_uav_ppo.rollout import Adapter, collect_slot


def output_directory(root, mode, seed):
    root = Path(root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
    output = root / 'random_uav_ppo' / f'seed_{seed}' / f'{mode}_{stamp}'
    output.mkdir(parents=True)
    print(f'Output directory: {output}', flush=True)
    return output


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    env = build_scene(settings_from_snapshot(checkpoint['environment']), episode=0)
    members = Adapter(env).policies(checkpoint['config']['hidden'], device)
    members.load_state_dict(checkpoint['member_policies'])
    members.eval()
    return env, members, checkpoint


def train(*, episodes=500, slots=None, users=None, dag_nodes=None, seed=45,
          device='cpu', hidden=128, rollout_slots=8, ppo_epochs=4, minibatch_size=256,
          learning_rate=3e-4, member_gamma=1., gae_lambda=.95, clip_ratio=.2,
          entropy_coef=.01, value_coef=.5, max_grad_norm=.5, torch_threads=1,
          output_root='results', diagnostics_interval=5, log_interval=5, snapshot_interval=10, metrics_interval=10, resample_slots=10):
    flight_seed = derive_flight_seed(seed)
    # 保存派生飞行种子供已有高层基线读取；它不是可配置的独立种子。
    config = dict(episodes=episodes, slots=slots, users=users, dag_nodes=dag_nodes, seed=seed,
        device=device, hidden=hidden, rollout_slots=rollout_slots,
        ppo_epochs=ppo_epochs, minibatch_size=minibatch_size, learning_rate=learning_rate,
        member_gamma=member_gamma, gae_lambda=gae_lambda, clip_ratio=clip_ratio,
        entropy_coef=entropy_coef, value_coef=value_coef, max_grad_norm=max_grad_norm,
        torch_threads=torch_threads, output_root=str(output_root),
        diagnostics_interval=diagnostics_interval, log_interval=log_interval,
        snapshot_interval=snapshot_interval, metrics_interval=metrics_interval, flight_seed=flight_seed, resample_slots=resample_slots)
    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    settings = resolved_settings(users, dag_nodes)
    if slots is not None:
        settings['env'] = replace(settings['env'], steps_per_epoch=slots)
    env = build_scene(settings, episode=0)
    adapter = Adapter(env, flight_seed=flight_seed, resample_slots=resample_slots)
    members = adapter.policies(hidden, device)
    optimizers = [torch.optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5) for policy in members]
    output = output_directory(output_root, 'train', seed)
    write_json(output / 'trajectory.json', adapter.flight.snapshot())
    snapshot = {name: asdict(value) for name, value in settings.items()}
    write_json(output / 'config.json', dict(method='random_uav_ppo', training=config,
        environment=snapshot, flight_policy='random_vector', ordering='ERS',
        member_reward='environment.member_rewards', member_terminal='end of scheduling slot',
        architecture='independent Member PPO; local actor and critic; no Master policy',
        policy_inputs='local member observation excluding energy queues',
        logging=dict(empty='undefined, no samples, or detailed diagnostics not collected',
                     indexing='episode/slot zero-based; update one-based; update slot exclusive end',
                     returns='GAE targets, not episode reward sums',
                     gradients='pre-clipping norms',
                     detailed='every diagnostics_interval updates and episode end')))
    write_json(output / 'scene.json', dict(member_positions=env.members.positions.tolist(),
        master_positions=env.masters.positions.tolist(), user_positions=env.users.positions.tolist(),
        member_region_ids=env.members.region_ids.tolist(), env_seed=env.env_config.env_seed,
        training_epochs=[0, episodes - 1], evaluation_epoch_start=1000000))
    metadata = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
        branch=git_output('branch', '--show-current'), commit=git_output('rev-parse', 'HEAD'),
        torch=str(torch.__version__), device=device)
    write_json(output / 'metadata.json', metadata)
    summary, updates, last_update = MetricSummary(), 0, {}

    def save_checkpoint(path, completed):
        checkpoint = dict(method='random_uav_ppo', config=config, environment=snapshot,
            episodes_completed=completed, updates=updates, last_update=last_update,
            member_policies=members.state_dict(), member_optimizers=[opt.state_dict() for opt in optimizers])
        temporary = path.with_suffix('.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    options = dict(ppo_epochs=ppo_epochs, minibatch_size=minibatch_size, clip_ratio=clip_ratio,
                   entropy_coef=entropy_coef, value_coef=value_coef, max_grad_norm=max_grad_norm)
    started = perf_counter()
    print(f'Random-flight PPO | Scene seed {env.env_config.env_seed} | Policy seed {seed} | '
          f'{episodes} episodes | {env.env_config.steps_per_epoch} slots/episode', flush=True)
    try:
        save_checkpoint(output / 'initial.pt', 0)
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as metric_file, \
                (output / 'updates.csv').open('w', newline='', encoding='utf-8') as update_file, \
                (output / 'episodes.csv').open('w', newline='', encoding='utf-8') as episode_file:
            metric_writer = update_writer = episode_writer = None
            for episode in range(episodes):
                env.reset(epoch_index=episode)
                episode_summary = MetricSummary()
                for start in range(0, env.env_config.steps_per_epoch, rollout_slots):
                    rollout_started = perf_counter()
                    member_samples = [[] for _ in members]
                    stop = min(start + rollout_slots, env.env_config.steps_per_epoch)
                    for slot in range(start, stop):
                        low, row = collect_slot(adapter, members, member_gamma=member_gamma, gae_lambda=gae_lambda)
                        for owner, samples in low.items():
                            member_samples[owner].extend(samples)
                        summary.add(row)
                        episode_summary.add(row)
                        if metric_writer is None:
                            metric_writer = csv.DictWriter(metric_file, fieldnames=list(row))
                            metric_writer.writeheader()
                        if (slot + 1) % metrics_interval == 0 or slot + 1 == env.env_config.steps_per_epoch:
                            metric_writer.writerow(row)
                    rollout_elapsed = perf_counter() - rollout_started
                    update_started = perf_counter()
                    detailed = (updates + 1) % diagnostics_interval == 0 or stop == env.env_config.steps_per_epoch
                    last_update = {}
                    for owner, (policy, optimizer, samples) in enumerate(zip(members, optimizers, member_samples)):
                        stats = update(policy, optimizer, samples, diagnostics=detailed, **options)
                        last_update.update({f'member_{owner}_{key}': value for key, value in stats.items()})
                    updates += 1
                    update_row = dict(update=updates, episode=episode, slot=stop,
                        rollout_elapsed_s=rollout_elapsed, update_elapsed_s=perf_counter() - update_started, **last_update)
                    if update_writer is None:
                        update_writer = csv.DictWriter(update_file, fieldnames=list(update_row))
                        update_writer.writeheader()
                    update_writer.writerow(update_row)
                    if updates % log_interval == 0 or stop == env.env_config.steps_per_epoch:
                        counts = [len(samples) for samples in member_samples]
                        print(f'Update {updates} | Episode {episode + 1}/{episodes} Slot {stop} | '
                              f'Delay {episode_summary.result()["mean_delay_s"]:.3f}s | '
                              f'Member samples {min(counts)}..{max(counts)} inactive {counts.count(0)}', flush=True)
                episode_row = dict(episode=episode, update=updates, **episode_summary.result())
                if episode_writer is None:
                    episode_writer = csv.DictWriter(episode_file, fieldnames=list(episode_row))
                    episode_writer.writeheader()
                episode_writer.writerow(episode_row)
                metric_file.flush()
                update_file.flush()
                episode_file.flush()
                save_checkpoint(output / 'checkpoint.pt', episode + 1)
                if episode == 0 or (episode + 1) % snapshot_interval == 0:
                    save_checkpoint(output / 'snapshot.pt', episode + 1)
        write_json(output / 'summary.json', dict(**summary.result(), updates=updates, elapsed_s=perf_counter() - started))
        metadata['status'] = 'completed'
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        metadata.update(finished_at=datetime.now(timezone.utc).isoformat(), slots_completed=summary.slots, updates=updates)
        write_json(output / 'metadata.json', metadata)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'config' / 'methods' / 'random_uav_ppo.toml')
    for option in ('episodes', 'slots', 'users', 'dag-nodes', 'seed', 'hidden',
                   'rollout-slots', 'ppo-epochs', 'minibatch-size', 'torch-threads',
                   'diagnostics-interval', 'metrics-interval', 'log-interval', 'snapshot-interval', 'resample-slots'):
        parser.add_argument('--' + option, type=int)
    for option in ('learning-rate', 'member-gamma', 'gae-lambda', 'clip-ratio',
                   'entropy-coef', 'value-coef', 'max-grad-norm'):
        parser.add_argument('--' + option, type=float)
    parser.add_argument('--device')
    parser.add_argument('--output-root')
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    with config_path.open('rb') as stream:
        config = tomllib.load(stream)['train']
    config.update({key: value for key, value in vars(args).items() if key != 'config' and value is not None})
    print(f'Completed: {train(**config)}', flush=True)


if __name__ == '__main__':
    main()
