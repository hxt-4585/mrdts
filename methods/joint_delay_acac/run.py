"""Joint online training with persistent CPU samplers and synchronized policies."""
import argparse
import csv
from datetime import datetime, timezone
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import traceback
import warnings

import torch

from methods.joint_delay_acac.learning import make_model, make_optimizers, update
from methods.joint_delay_acac.planner import Controller, observe
from methods.joint_delay_acac.rollout import collect_slot
from methods.joint_delay_acac.support import (ROOT, aggregate, cpu_weights, environment_snapshot,
                                       load_config, make_environment, replace_file, write_json)


def _sample_worker(connection, snapshot, config, index):
    try:
        torch.set_num_threads(1)
        torch.manual_seed(config['train']['seed'] + 200 + index)
        env = make_environment(snapshot)
        members = env.members.member_uav_count
        model = make_model(members, 5 * members + 8, config['train'], 'cpu')
        model.eval()
        controller = Controller(observe(env), config['high'])
        epoch = None
        while True:
            job = connection.recv()
            if job is None:
                break
            model.load_state_dict(job['low'])
            controller.load_policy(job['high'])
            if epoch != job['epoch'] or job['evaluation']:
                epoch = job['epoch']
                env.reset(epoch_index=epoch)
                controller.reset(epoch)
            trajectories, rows, experiences = [], [], []
            for _ in range(job['slots']):
                views = observe(env)
                started = time.perf_counter()
                actions = controller.actions(views, job['exploration'])
                planning_s = time.perf_counter() - started
                trajectory, row, regional = collect_slot(
                    env, model, actions, config['train'], deterministic=job['evaluation'])
                row.update(epoch=epoch, slot=env.step_index - 1, planning_s=planning_s)
                rows.append(row)
                if not job['evaluation']:
                    trajectories.append(trajectory)
                    experiences.append(controller.experience(views, actions, regional))
            connection.send(('ok', (trajectories, rows, experiences)))
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


class Samplers:
    """One pipe per persistent process keeps each environment on its own worker."""
    def __init__(self, snapshot, config, workers):
        self.channels, self.processes = [], []
        context = mp.get_context('spawn')
        for index in range(workers):
            parent, child = context.Pipe()
            process = context.Process(target=_sample_worker, args=(child, snapshot, config, index))
            process.start()
            child.close()
            self.channels.append(parent)
            self.processes.append(process)

    def sample(self, jobs, low, high):
        active = []
        for index, job in enumerate(jobs):
            if job is not None:
                self.channels[index].send(dict(job, low=low, high=high))
                active.append(index)
        results = []
        for index in active:
            status, result = self.channels[index].recv()
            if status == 'error':
                raise RuntimeError(result)
            results.append(result)
        return results

    def close(self):
        for channel, process in zip(self.channels, self.processes):
            if process.is_alive():
                try:
                    channel.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for channel, process in zip(self.channels, self.processes):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
            channel.close()


def evaluate(checkpoint, epochs, *, slots=None, workers=None):
    """Frozen deterministic paired evaluation: no replay insertion or updates."""
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    config, snapshot = saved['config'], saved['environment']
    slots = slots if slots is not None else config['train']['evaluation_slots']
    worker_count = min(workers or config['train']['workers'], len(epochs))
    high = [(state['model'], state['samples']) for state in saved['high']]
    pool = Samplers(snapshot, config, worker_count)
    episodes = []
    try:
        for offset in range(0, len(epochs), worker_count):
            jobs = [dict(epoch=epoch, slots=slots, evaluation=True, exploration=0.)
                    for epoch in epochs[offset:offset + worker_count]]
            for _, rows, _ in pool.sample(jobs, saved['low'], high):
                episodes.append(dict(epoch=rows[0]['epoch'], **aggregate(rows),
                                     planning_mean_s=sum(r['planning_s'] for r in rows) / len(rows)))
    finally:
        pool.close()
    return episodes


def slot_quotas(remaining, budget, offset):
    """Distribute a fixed global budget over unfinished sampler streams."""
    active = [i for i, slots in enumerate(remaining) if slots]
    rotation = offset % len(active)
    active = active[rotation:] + active[:rotation]
    quotas = [0] * len(remaining)
    cursor = 0
    for _ in range(min(budget, sum(remaining))):
        position = cursor % len(active)
        owner = active[position]
        quotas[owner] += 1
        if quotas[owner] == remaining[owner]:
            active.pop(position)
            cursor = position
        else:
            cursor += 1
    return quotas


def copy_completed_history(source, output, completed, updates):
    """Keep only records represented by the restored completed-episode checkpoint."""
    elapsed = 0.
    for name, counter, limit in (('episodes.csv', 'episode', completed),
                                 ('updates.csv', 'update', updates)):
        path = source / name
        if path.exists():
            with path.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = [row for row in reader if int(row[counter]) <= limit]
            with (output / name).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            if name == 'episodes.csv' and rows:
                elapsed = float(rows[-1]['elapsed_s'])
    for path in source.glob('validation_*.json'):
        if int(path.stem.split('_')[1]) <= completed:
            shutil.copyfile(path, output / path.name)
    for name in ('initial.pt', 'best.pt'):
        shutil.copyfile(source / name, output / name)
    return elapsed


def train(config, output, *, resume=None):
    train_config = config['train']
    torch.set_num_threads(1)
    torch.manual_seed(train_config['seed'])
    output = Path(output).resolve()
    output.mkdir(parents=True)
    restored = torch.load(resume, weights_only=True, map_location='cpu') if resume is not None else None
    snapshot = restored['environment'] if restored is not None else environment_snapshot(config)
    env = make_environment(snapshot)
    count = env.members.member_uav_count
    low = make_model(count, 5 * count + 8, train_config, train_config['device'])
    optimizers = make_optimizers(low, train_config)
    high = Controller(observe(env), config['high'])
    started = time.perf_counter()
    completed = sampled = updates = 0
    best, selected = float('inf'), 0
    previous_elapsed = 0.
    if restored is not None:
        low.load_state_dict(restored['low'])
        optimizer_list = optimizers if isinstance(optimizers, list) else [optimizers]
        for optimizer, state in zip(optimizer_list, restored['low_optimizers']):
            optimizer.load_state_dict(state)
        for model, state in zip(high.models, restored['high']):
            model.restore(state)
        high.observations = restored['high_observations']
        high.generator.set_state(restored['high_rng'])
        completed, sampled, updates = (restored['episodes_completed'], restored['slots_completed'], restored['updates'])
        best, selected = restored['best_validation_delay_s'], restored['selected_episode']
        previous_elapsed = copy_completed_history(Path(resume).resolve().parent, output, completed, updates)
        selection = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
        best, selected = selection['best_validation_delay_s'], selection['selected_episode']
        torch.set_rng_state(restored['torch_rng'])
        if train_config['device'].startswith('cuda'):
            torch.cuda.set_rng_state(restored['cuda_rng'])
    metadata = dict(status='running', pid=os.getpid(), phase='initialization',
                    episodes_completed=0, slots_completed=0, updates=0,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    device=train_config['device'], workers=train_config['workers'])
    if restored is not None:
        metadata.update(resumed_from=str(Path(resume).resolve()), previous_training_elapsed_s=previous_elapsed,
                        recovery='completed-episode boundary; sampler random streams restarted',
                        best_validation_delay_s=best, selected_episode=selected)
    write_json(output / 'config.json', dict(method='joint_delay_acac', config=config, environment=snapshot,
        initialization='resumed joint checkpoint' if restored is not None else 'both levels from scratch; no pretrained checkpoint',
        resumed_from=str(Path(resume).resolve()) if restored is not None else None,
        training='fixed policy per rollout; central low/high updates; synchronized CPU samplers',
        time='(decision time - scheduling slot start) / nominal scheduling duration',
        discount='PPO: agent decisions; ACAC: gamma by physical time, lambda by joint event',
        high_update='one global feedback sample per sampled slot, same replay and fit budget in both groups',
        limitation='joint online learning, not a differentiable gradient through the simulator'))

    last_status_write = -float('inf')

    def status(**changes):
        nonlocal last_status_write
        now = time.perf_counter()
        metadata.update(changes, episodes_completed=completed, slots_completed=sampled, updates=updates,
                        elapsed_s=previous_elapsed + now - started,
                        updated_at=datetime.now(timezone.utc).isoformat())
        force = metadata['phase'] in ('initialization', 'validation', 'test', 'done', 'failed')
        if force or now - last_status_write >= 5.:
            last_status_write = now
            try:
                write_json(output / 'metadata.json', metadata)
            except PermissionError as error:
                warnings.warn(f'Progress file remains locked; training continues: {error}', RuntimeWarning)

    def save(name):
        opts = optimizers if isinstance(optimizers, list) else [optimizers]
        state = dict(method='joint_delay_acac', config=config, environment=snapshot, low=cpu_weights(low),
                     low_optimizers=[optimizer.state_dict() for optimizer in opts],
                     high=[model.snapshot() for model in high.models],
                     high_observations=high.observations, high_rng=high.generator.get_state(),
                     episodes_completed=completed, slots_completed=sampled, updates=updates,
                     best_validation_delay_s=best, selected_episode=selected,
                     torch_rng=torch.get_rng_state())
        if train_config['device'].startswith('cuda'):
            state['cuda_rng'] = torch.cuda.get_rng_state()
        pending = output / (name + '.tmp')
        torch.save(state, pending)
        replace_file(pending, output / name)

    def validate():
        nonlocal best, selected
        status(phase='validation')
        save('candidate.pt')
        episodes = evaluate(output / 'candidate.pt', train_config['validation_epochs'])
        delay = sum(row['mean_delay_s'] for row in episodes) / len(episodes)
        write_json(output / f'validation_{completed:04d}.json', episodes)
        if delay < best:
            best, selected = delay, completed
            save('best.pt')
        status(best_validation_delay_s=best, selected_episode=selected)
        return delay

    pool = None
    try:
        status()
        if restored is None:
            save('initial.pt')
            validate()
        else:
            save('resume_initial.pt')
        save('latest.pt')
        pool = Samplers(snapshot, config, train_config['workers'])
        next_validation = (completed // train_config['validation_interval'] + 1) * train_config['validation_interval']
        update_path, episode_path = output / 'updates.csv', output / 'episodes.csv'
        with update_path.open('a', encoding='utf-8', newline='') as update_stream, \
             episode_path.open('a', encoding='utf-8', newline='') as episode_stream:
            update_writer = episode_writer = None
            if restored is not None:
                for path, label in ((update_path, 'updates'), (episode_path, 'episodes')):
                    with path.open(encoding='utf-8', newline='') as stream:
                        fields = next(csv.reader(stream))
                    if label == 'updates':
                        update_writer = csv.DictWriter(update_stream, fieldnames=fields)
                    else:
                        episode_writer = csv.DictWriter(episode_stream, fieldnames=fields)
            for group_start in range(completed, train_config['episodes'], train_config['workers']):
                group_size = min(train_config['workers'], train_config['episodes'] - group_start)
                remaining = [train_config['slots']] * group_size
                histories = [[] for _ in range(group_size)]
                exploration_progress = min(1., group_start / 40.)
                exploration = (config['high']['exploration_start'] * (1 - exploration_progress) +
                               config['high']['exploration_end'] * exploration_progress)
                while any(remaining):
                    quotas = slot_quotas(remaining, train_config['rollout_slots'], updates)
                    jobs = []
                    for i in range(group_size):
                        quota = quotas[i]
                        jobs.append(dict(epoch=train_config['train_epoch_start'] + group_start + i,
                                         slots=quota, exploration=exploration, evaluation=False) if quota else None)
                    status(phase='sampling', active_episodes=list(range(group_start, group_start + group_size)))
                    tick = time.perf_counter()
                    batches = pool.sample(jobs, cpu_weights(low), high.policy())
                    sampling_s = time.perf_counter() - tick
                    trajectories, experiences, rows = [], [], []
                    result_index = 0
                    for i, job in enumerate(jobs):
                        if job is not None:
                            local_trajectories, local_rows, local_experiences = batches[result_index]
                            result_index += 1
                            histories[i].extend(local_rows)
                            remaining[i] -= job['slots']
                            trajectories.extend(local_trajectories)
                            rows.extend(local_rows)
                            experiences.extend(local_experiences)
                    status(phase='updating')
                    tick = time.perf_counter()
                    low_stats = update(low, optimizers, trajectories, train_config)
                    proxy_loss = high.feedback(experiences)
                    learning_s = time.perf_counter() - tick
                    sampled += len(rows)
                    updates += 1
                    row = dict(update=updates, slots_completed=sampled, **aggregate(rows),
                               sampling_s=sampling_s, learning_s=learning_s, proxy_loss=proxy_loss,
                               exploration=exploration, elapsed_s=previous_elapsed + time.perf_counter() - started,
                               **{f'low_{k}': v for k, v in low_stats.items()})
                    if update_writer is None:
                        update_writer = csv.DictWriter(update_stream, fieldnames=list(row))
                        update_writer.writeheader()
                    update_writer.writerow(row)
                    update_stream.flush()
                    status(phase='training', mean_delay_s=row['mean_delay_s'],
                           sampling_s=sampling_s, learning_s=learning_s, proxy_loss=proxy_loss)
                    if updates % train_config['checkpoint_interval'] == 0:
                        save('latest.pt')
                for i, rows in enumerate(histories):
                    row = dict(episode=group_start + i + 1,
                               epoch=train_config['train_epoch_start'] + group_start + i,
                               **aggregate(rows), updates=updates, elapsed_s=previous_elapsed + time.perf_counter() - started)
                    if episode_writer is None:
                        episode_writer = csv.DictWriter(episode_stream, fieldnames=list(row))
                        episode_writer.writeheader()
                    episode_writer.writerow(row)
                    print(row, flush=True)
                episode_stream.flush()
                completed += group_size
                save('boundary.pt')
                if completed >= next_validation or completed == train_config['episodes']:
                    validate()
                    next_validation = (completed // train_config['validation_interval'] + 1) * train_config['validation_interval']
                save('latest.pt')
                status(phase='training')
        pool.close()
        pool = None
        status(phase='test')
        episodes = evaluate(output / 'best.pt', train_config['test_epochs'])
        write_json(output / 'test_results.json', dict(episodes=episodes, selected_episode=selected,
            mean_delay_s=sum(row['mean_delay_s'] for row in episodes) / len(episodes),
            failure_rate=sum(row['failed_dags'] for row in episodes) / sum(row['dag_count'] for row in episodes)))
        status(status='completed', phase='done', finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status(status='failed', phase='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if pool is not None:
            pool.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', type=Path, help='Completed-episode checkpoint; writes continuation to a new directory')
    for name in ('episodes', 'slots', 'workers', 'users', 'dag-nodes', 'evaluation-slots'):
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    config = (torch.load(args.resume, map_location='cpu', weights_only=True)['config']
              if args.resume is not None else load_config(args.config))
    for name in ('episodes', 'slots', 'workers', 'users', 'dag_nodes', 'evaluation_slots', 'device'):
        value = getattr(args, name)
        if value is not None:
            config['train'][name] = value
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = args.output_dir or ROOT / config['train']['output_root'] / f'seed_{config["train"]["seed"]}' / stamp
    train(config, output, resume=args.resume)


if __name__ == '__main__':
    main()

