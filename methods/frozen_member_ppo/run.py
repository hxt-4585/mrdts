"""冻结低层的独立高层 PPO 基线。

python methods/frozen_member_ppo/run.py --config config/methods/frozen_member_ppo.toml
环境及网络宽度取自低层检查点；仅 slots 可覆盖，以便短程验证。
高层每时隙决策，跨时隙 GAE；低层始终 argmax，不创建低层优化器。
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ''):
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.random_uav_ppo.support import settings_from_snapshot
from methods.hmappo.ppo import finish_trajectory, update
from methods.hmappo.rollout import Adapter, collect_slot, master_bootstrap
from methods.hmappo.support import build_scene, MetricSummary, write_json, git_output


def output_directory(root, mode, seed):
    root = Path(root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
    output = root / 'frozen_member_ppo' / f'seed_{seed}' / f'{mode}_{stamp}'
    output.mkdir(parents=True)
    print(f'Output directory: {output}', flush=True)
    return output


def frozen_policies(env, hidden, weights, device):
    master, members = Adapter(env).policies(hidden, device)
    members.load_state_dict(weights)
    members.requires_grad_(False)
    members.eval()
    return master, members


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    env = build_scene(settings_from_snapshot(saved['environment']), episode=0)
    master, members = frozen_policies(env, saved['config']['hidden'], saved['member_policies'], device)
    master.load_state_dict(saved['master'])
    master.eval()
    return env, master, members, saved


def train(*, member_checkpoint, episodes=500, slots=None, seed=45, device='cpu',
          torch_threads=1, rollout_slots=64, ppo_epochs=4, minibatch_size=256,
          learning_rate=3e-4, master_gamma=.99, gae_lambda=.95, clip_ratio=.2,
          entropy_coef=.01, value_coef=.5, max_grad_norm=.5, boundary_penalty=1.,
          output_root='results', snapshot_interval=10):
    source_path = Path(member_checkpoint)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 同一份字节用于载入及指纹，记录本次实际使用的低层版本。
    source_bytes = source_path.read_bytes()
    source = torch.load(io.BytesIO(source_bytes), map_location=device, weights_only=True)
    member_source = dict(path=str(source_path.resolve()), sha256=hashlib.sha256(source_bytes).hexdigest(),
                         method=source['method'], episodes_completed=source['episodes_completed'])
    settings = settings_from_snapshot(source['environment'])
    if slots is not None:
        settings['env'] = replace(settings['env'], steps_per_epoch=slots)
    hidden = source['config']['hidden']
    config = dict(member_checkpoint=str(source_path.resolve()), episodes=episodes, slots=slots,
        seed=seed, device=device, torch_threads=torch_threads, hidden=hidden, rollout_slots=rollout_slots,
        ppo_epochs=ppo_epochs, minibatch_size=minibatch_size, learning_rate=learning_rate,
        master_gamma=master_gamma, gae_lambda=gae_lambda, clip_ratio=clip_ratio,
        entropy_coef=entropy_coef, value_coef=value_coef, max_grad_norm=max_grad_norm,
        boundary_penalty=boundary_penalty, output_root=str(output_root), snapshot_interval=snapshot_interval)
    flight_config = dict(flight_seed=source['config']['flight_seed'],
                         resample_slots=source['config']['resample_slots'])
    env = build_scene(settings, episode=0)
    adapter = Adapter(env)
    master, members = frozen_policies(env, hidden, source['member_policies'], device)
    del source, source_bytes
    optimizer = torch.optim.Adam(master.parameters(), lr=learning_rate, eps=1e-5)
    output = output_directory(output_root, 'train', seed)
    print(f'Frozen Member PPO | Source episode {member_source["episodes_completed"]} | '
          f'{episodes} episodes, {env.env_config.steps_per_epoch} slots/episode | '
          f'{adapter.masters} Masters, {adapter.members} frozen Members', flush=True)
    snapshot = {name: asdict(value) for name, value in settings.items()}
    write_json(output / 'config.json', dict(method='frozen_member_ppo', training=config,
        environment=snapshot, member_source=member_source, frozen_members=True, member_policy='argmax',
        architecture='shared Master actor, centralized critic; frozen independent Members by UAV ID',
        ordering='ERS', master_reward='environment.master_reward - boundary_penalty * owned_violations',
        evaluation_epoch_start=1000000, random_flight=flight_config,
        logging='one row per episode; episode_reward_sum sums all Master training rewards including boundary penalties'))
    metadata = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
        branch=git_output('branch', '--show-current'), commit=git_output('rev-parse', 'HEAD'),
        torch=str(torch.__version__), device=device)
    write_json(output / 'metadata.json', metadata)
    summary, updates, last_update = MetricSummary(), 0, {}

    def save(path, completed):
        saved = dict(method='frozen_member_ppo', config=config, environment=snapshot,
            member_source=member_source, random_flight=flight_config, episodes_completed=completed,
            updates=updates, last_update=last_update, master=master.state_dict(),
            member_policies=members.state_dict(), master_optimizer=optimizer.state_dict())
        temporary = path.with_suffix('.tmp')
        torch.save(saved, temporary)
        temporary.replace(path)

    options = dict(ppo_epochs=ppo_epochs, minibatch_size=minibatch_size, clip_ratio=clip_ratio,
        entropy_coef=entropy_coef, value_coef=value_coef, max_grad_norm=max_grad_norm)
    started = perf_counter()
    try:
        save(output / 'initial.pt', 0)
        with (output / 'episodes.csv').open('w', newline='', encoding='utf-8') as episodes_file:
            episode_writer = None
            for episode in range(episodes):
                episode_started = perf_counter()
                env.reset(epoch_index=episode)
                episode_summary = MetricSummary()
                master_reward_sums = np.zeros(adapter.masters)
                for start in range(0, env.env_config.steps_per_epoch, rollout_slots):
                    rollout_started = perf_counter()
                    trajectories = [[] for _ in range(adapter.masters)]
                    stop = min(start + rollout_slots, env.env_config.steps_per_epoch)
                    for _ in range(start, stop):
                        high, _, row = collect_slot(adapter, master, members,
                            member_deterministic=True, boundary_penalty=boundary_penalty)
                        for owner, (trajectory, sample) in enumerate(zip(trajectories, high)):
                            trajectory.append(sample)
                            master_reward_sums[owner] += sample.reward
                        summary.add(row)
                        episode_summary.add(row)
                    bootstrap = (np.zeros(adapter.masters) if stop == env.env_config.steps_per_epoch
                                 else master_bootstrap(adapter, master))
                    samples = [sample for trajectory, value in zip(trajectories, bootstrap)
                               for sample in finish_trajectory(trajectory, value, master_gamma, gae_lambda)]
                    rollout_elapsed = perf_counter() - rollout_started
                    update_started = perf_counter()
                    stats = update(master, optimizer, samples, **options)
                    updates += 1
                    last_update = dict(update=updates, episode=episode, slot=stop,
                        rollout_elapsed_s=rollout_elapsed, update_elapsed_s=perf_counter() - update_started,
                        bootstrap_abs_mean=float(np.abs(bootstrap).mean()),
                        **{f'master_{key}': value for key, value in stats.items()})
                row = dict(episode=episode, update=updates,
                    episode_reward_sum=float(master_reward_sums.sum()),
                    **{f'master_{i}_reward_sum': float(value) for i, value in enumerate(master_reward_sums)},
                    **episode_summary.result(), elapsed_s=perf_counter() - episode_started)
                if episode_writer is None:
                    episode_writer = csv.DictWriter(episodes_file, fieldnames=list(row))
                    episode_writer.writeheader()
                episode_writer.writerow(row)
                episodes_file.flush()
                print(f'Episode {episode + 1}/{episodes} | Reward sum {row["episode_reward_sum"]:.4f} | '
                      f'Mean delay {row["mean_delay_s"]:.4f}s', flush=True)
                save(output / 'checkpoint.pt', episode + 1)
                if episode == 0 or (episode + 1) % snapshot_interval == 0:
                    save(output / 'snapshot.pt', episode + 1)
        write_json(output / 'summary.json', dict(**summary.result(), updates=updates,
                                               elapsed_s=perf_counter() - started))
        metadata['status'] = 'completed'
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        metadata.update(finished_at=datetime.now(timezone.utc).isoformat(), slots_completed=summary.slots,
                        updates=updates)
        write_json(output / 'metadata.json', metadata)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'config/methods/frozen_member_ppo.toml')
    parser.add_argument('--member-checkpoint', type=Path)
    for name in ('episodes', 'slots', 'seed', 'torch-threads', 'rollout-slots', 'ppo-epochs',
                 'minibatch-size', 'snapshot-interval'):
        parser.add_argument('--' + name, type=int)
    for name in ('learning-rate', 'master-gamma', 'gae-lambda', 'clip-ratio', 'entropy-coef',
                 'value-coef', 'max-grad-norm', 'boundary-penalty'):
        parser.add_argument('--' + name, type=float)
    parser.add_argument('--device')
    parser.add_argument('--output-root')
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    with path.open('rb') as stream:
        config = tomllib.load(stream)['train']
    config.update({k: v for k, v in vars(args).items() if k != 'config' and v is not None})
    print(f'Completed: {train(**config)}', flush=True)


if __name__ == '__main__':
    main()
