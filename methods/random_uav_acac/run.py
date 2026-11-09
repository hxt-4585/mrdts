"""Train the independent random-flight ACAC adaptation on the shared MRDTS env.

python methods/random_uav_acac/run.py --config config/methods/random_uav_acac.toml
Actor inputs, reward and terminals match random_uav_ppo. Unlike the paper's
single team value, the attention critic predicts one value per Member. Value
loss uses raw returns, matching the PPO reference (no PopArt normalization).
"""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from time import perf_counter
import tomllib
import uuid

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from methods.random_uav_acac.learning import update
from methods.random_uav_acac.network import ACAC
from methods.random_uav_acac.rollout import Adapter, collect_slot
from methods.random_uav_acac.scene import (
    ROOT, Summary, build_scene, resolved_settings, settings_from_snapshot, write_json,
)


class Rows:
    def __init__(self, stream):
        self.stream, self.writer = stream, None

    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)


def output_directory(root, mode, seed):
    root = Path(root)
    if not root.is_absolute():
        root = ROOT / root
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path = root / 'random_uav_acac' / f'seed_{seed}' / f'{mode}_{tag}_{uuid.uuid4().hex[:8]}'
    path.mkdir(parents=True)
    print(f'Output directory: {path}', flush=True)
    return path


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    env = build_scene(settings_from_snapshot(saved['environment']), episode=0)
    config = saved['config']
    adapter = Adapter(env, config['seed'], config['resample_slots'])
    model = ACAC(adapter.obs_dim, adapter.count, config['hidden'],
                 config['time_dim'], config['attention_heads']).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    return env, model, saved


def train(*, episodes=500, slots=None, users=None, dag_nodes=None, seed=45, device='cpu',
          torch_threads=1, hidden=128, rollout_slots=8, ppo_epochs=4, minibatch_size=256,
          learning_rate=.0003, member_gamma=1., gae_lambda=.95, clip_ratio=.2,
          entropy_coef=.01, value_coef=.5, max_grad_norm=.5, output_root='results',
          diagnostics_interval=5, log_interval=5, metrics_interval=10, snapshot_interval=10,
          resample_slots=10, time_dim=16, attention_heads=4):
    config = dict(locals())
    config['output_root'] = str(output_root)
    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    settings = resolved_settings(users, dag_nodes)
    if slots is not None:
        settings['env'] = replace(settings['env'], steps_per_epoch=slots)
    snapshot = {name: asdict(value) for name, value in settings.items()}
    env = build_scene(settings, episode=0)
    adapter = Adapter(env, seed, resample_slots)
    model = ACAC(adapter.obs_dim, adapter.count, hidden, time_dim, attention_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    config['flight_seed'] = adapter.flight.seed
    output = output_directory(output_root, 'train', seed)
    write_json(output / 'config.json', dict(method='random_uav_acac', training=config, environment=snapshot,
        source='https://proceedings.mlr.press/v267/jung25a.html',
        adaptation='per-Member value outputs and original Member rewards; raw value loss without PopArt',
        architecture='independent time/GRU actors; separate per-agent critic encoders; self-attention and mean pooling',
        ordering='ERS', terminal='end of scheduling slot; histories reset and zero bootstrap',
        time='(runtime.now - slot_start) / scheduling_duration_s',
        gae='gamma**elapsed_time, lambda once per joint decision event; completion-time reward impulses',
        minibatch='whole slots, soft cap on decision count; full recurrent replay under current parameters',
        diagnostics='aggregate losses each update; per-Member rollout stats every diagnostics_interval and episode end'))
    write_json(output / 'trajectory.json', adapter.flight.snapshot())
    write_json(output / 'scene.json', dict(user_positions=env.users.positions.tolist(),
        member_positions=env.members.positions.tolist(), master_positions=env.masters.positions.tolist(),
        env_seed=env.env_config.env_seed, training_epochs=[0, episodes - 1], evaluation_epoch_start=1000000))
    metadata = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                    device=device, torch=str(torch.__version__))
    metadata['source_sha256'] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in (ROOT / 'env', Path(__file__).resolve().parent)
        for path in sorted(directory.rglob('*.py'))}
    write_json(output / 'metadata.json', metadata)
    total, updates = Summary(), 0
    options = dict(ppo_epochs=ppo_epochs, minibatch_size=minibatch_size, clip_ratio=clip_ratio,
                   entropy_coef=entropy_coef, value_coef=value_coef, max_grad_norm=max_grad_norm)

    def save(name, completed):
        path = output / name
        temporary = path.with_suffix('.tmp')
        torch.save(dict(method='random_uav_acac', config=config, environment=snapshot,
                        model=model.state_dict(), optimizer=optimizer.state_dict(),
                        episodes_completed=completed, updates=updates), temporary)
        temporary.replace(path)

    started = perf_counter()
    print(f'Random-flight ACAC | Scene seed {env.env_config.env_seed} | Policy seed {seed} | '
          f'{episodes} episodes | {env.env_config.steps_per_epoch} slots/episode', flush=True)
    try:
        save('initial.pt', 0)
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as mf, \
                (output / 'updates.csv').open('w', newline='', encoding='utf-8') as uf, \
                (output / 'episodes.csv').open('w', newline='', encoding='utf-8') as ef:
            metrics, update_rows, episode_rows = Rows(mf), Rows(uf), Rows(ef)
            for episode in range(episodes):
                env.reset(epoch_index=episode)
                summary = Summary()
                for start in range(0, env.env_config.steps_per_epoch, rollout_slots):
                    stop = min(start + rollout_slots, env.env_config.steps_per_epoch)
                    rollout_start, trajectories = perf_counter(), []
                    for slot in range(start, stop):
                        trajectory, row = collect_slot(adapter, model, member_gamma=member_gamma, gae_lambda=gae_lambda)
                        trajectories.append(trajectory)
                        total.add(row)
                        summary.add(row)
                        if (slot + 1) % metrics_interval == 0 or slot + 1 == env.env_config.steps_per_epoch:
                            metrics.add(row)
                    rollout_elapsed = perf_counter() - rollout_start
                    update_start = perf_counter()
                    stats = update(model, optimizer, trajectories, **options)
                    updates += 1
                    detailed = updates % diagnostics_interval == 0 or stop == env.env_config.steps_per_epoch
                    for owner in range(adapter.count):
                        local = [(trajectory, index) for trajectory in trajectories
                                 for index, event in enumerate(trajectory.events) if owner in event.owners]
                        stats[f'member_{owner}_samples'] = len(local)
                        for name, field in (('advantage_mean', 'advantages'), ('return_mean', 'targets'),
                                            ('value_mean', 'values')):
                            stats[f'member_{owner}_{name}'] = (float(np.mean([
                                getattr(trajectory, field)[index, owner] for trajectory, index in local]))
                                if detailed and local else None)
                    update_rows.add(dict(update=updates, episode=episode, slot=stop,
                        rollout_elapsed_s=rollout_elapsed, update_elapsed_s=perf_counter() - update_start, **stats))
                    if updates % log_interval == 0 or stop == env.env_config.steps_per_epoch:
                        print(f'Update {updates} | Episode {episode + 1}/{episodes} Slot {stop} | '
                              f'Delay {summary.result()["mean_delay_s"]:.4f}s | Samples {stats["samples"]}', flush=True)
                episode_rows.add(dict(episode=episode, update=updates, **summary.result()))
                for stream in (mf, uf, ef):
                    stream.flush()
                save('checkpoint.pt', episode + 1)
                if episode == 0 or (episode + 1) % snapshot_interval == 0:
                    save('snapshot.pt', episode + 1)
        write_json(output / 'summary.json', dict(**total.result(), updates=updates, elapsed_s=perf_counter() - started))
        metadata['status'] = 'completed'
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        metadata.update(finished_at=datetime.now(timezone.utc).isoformat(),
                        slots_completed=total.slots, updates=updates, elapsed_s=perf_counter() - started)
        write_json(output / 'metadata.json', metadata)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/methods/random_uav_acac.toml')
    for name in ('episodes', 'slots', 'users', 'dag-nodes', 'seed', 'torch-threads', 'hidden',
                 'rollout-slots', 'ppo-epochs', 'minibatch-size', 'diagnostics-interval', 'log-interval',
                 'metrics-interval', 'snapshot-interval', 'resample-slots', 'time-dim', 'attention-heads'):
        parser.add_argument('--' + name, type=int)
    for name in ('learning-rate', 'member-gamma', 'gae-lambda', 'clip-ratio', 'entropy-coef',
                 'value-coef', 'max-grad-norm'):
        parser.add_argument('--' + name, type=float)
    parser.add_argument('--device')
    parser.add_argument('--output-root')
    args = vars(parser.parse_args())
    path = args.pop('config')
    with (path if path.is_absolute() else ROOT / path).open('rb') as stream:
        config = tomllib.load(stream)['train']
    config.update({name: value for name, value in args.items() if value is not None})
    print(f'Completed: {train(**config)}', flush=True)


if __name__ == '__main__':
    main()
