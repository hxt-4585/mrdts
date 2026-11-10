"""Train with persistent CPU samplers; checkpoint at completed episode boundaries.

Run: python -m methods.mrdts_delay.run --config config/methods/mrdts_delay.toml
Resume: python -m methods.mrdts_delay.run --resume results/.../last.pt --episodes 125
The slots and rollout_slots budgets are TOTAL across all sampler processes.
"""
import argparse
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import torch

from methods.mrdts_delay.high import DelayPlanner
from methods.mrdts_delay.low import PPO
from methods.mrdts_delay.runtime import build_environment, default_settings, run_slot
from methods.mrdts_delay.storage import (append_rows, load_config, pack_settings, save_checkpoint,
                                        summarize, unpack_settings, write_json)


def _worker(pipe, snapshot, config, index, rng):
    try:
        torch.set_num_threads(1)
        torch.manual_seed(config['train']['seed'] + 200 + index)
        env = build_environment(unpack_settings(snapshot))
        low = PPO(env.members.member_uav_count, config['train'], device='cpu')
        planner = DelayPlanner(env, config['high'])
        if rng is not None:
            torch.set_rng_state(rng)
        active_epoch = None
        while True:
            job = pipe.recv()
            if job is None:
                return
            low.policies.load_state_dict(job['low'])
            planner.load_policy(job['high'])
            if active_epoch != job['epoch'] or job['evaluation']:
                active_epoch = job['epoch']
                env.reset(epoch_index=active_epoch)
                planner.reset(active_epoch)
            trajectories, rows, experiences = [], [], []
            for _ in range(job['slots']):
                views = planner.views(env)
                started = time.perf_counter()
                actions = planner.act(views, job['exploration'])
                planning_s = time.perf_counter() - started
                samples, row, targets = run_slot(env, low, actions, config['train'], job['evaluation'])
                row.update(worker=index, planning_s=planning_s)
                rows.append(row)
                if not job['evaluation']:
                    trajectories.append(samples)
                    experiences.append(planner.experience(views, actions, targets))
            pipe.send(dict(trajectories=trajectories, rows=rows, experiences=experiences,
                           rng=torch.get_rng_state()))
    except BaseException:
        pipe.send(dict(error=traceback.format_exc()))
    finally:
        pipe.close()


class SamplerPool:
    def __init__(self, snapshot, config, count, rngs=None):
        self.pipes, self.processes = [], []
        self.rngs = list(rngs) if rngs is not None else [None] * count
        context = mp.get_context('spawn')
        for index in range(count):
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, snapshot, config, index, self.rngs[index]))
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)

    def collect(self, jobs, low, high):
        for index, job in jobs:
            self.pipes[index].send(dict(job, low=low, high=high))
        results = []
        for index, _ in jobs:
            result = self.pipes[index].recv()
            if 'error' in result:
                raise RuntimeError(result['error'])
            self.rngs[index] = result['rng']
            results.append(result)
        return results

    def close(self):
        for pipe, process in zip(self.pipes, self.processes):
            if process.is_alive():
                try:
                    pipe.send(None)
                except (BrokenPipeError, OSError):
                    pass
        for pipe, process in zip(self.pipes, self.processes):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
            pipe.close()


def _evaluate_state(state, epochs, slots, workers):
    config = state['config']
    pool = SamplerPool(state['environment'], config, min(workers, len(epochs)))
    high = [(model['weights'], model['size']) for model in state['high']['models']]
    rows = []
    try:
        for offset in range(0, len(epochs), len(pool.pipes)):
            jobs = [(i, dict(epoch=epoch, slots=slots, evaluation=True, exploration=0.))
                    for i, epoch in enumerate(epochs[offset:offset + len(pool.pipes)])]
            for result in pool.collect(jobs, state['low']['weights'], high):
                rows.extend(result['rows'])
    finally:
        pool.close()
    return dict(summary=summarize(rows), epochs=list(epochs), rows=rows)


def evaluate(checkpoint, epochs=None, *, slots=None, workers=None):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = state['config']['train']
    return _evaluate_state(state, epochs if epochs is not None else cfg['test_epochs'],
                           slots if slots is not None else cfg['evaluation_slots'],
                           workers if workers is not None else cfg['workers'])


def train(config, output, *, resume=None, episodes=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(resume, map_location='cpu', weights_only=True) if resume is not None else None
    if saved is not None:
        config = saved['config']
    if episodes is not None:
        config['train']['episodes'] = episodes
    cfg = config['train']
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    settings = unpack_settings(saved['environment']) if saved else default_settings(cfg)
    snapshot = pack_settings(settings)
    env = build_environment(settings)
    low = PPO(env.members.member_uav_count, cfg)
    planner = DelayPlanner(env, config['high'])
    completed = total_slots = 0
    best = float('inf')
    if saved:
        low.restore(saved['low'])
        planner.restore(saved['high'])
        completed, total_slots = saved['completed_episodes'], saved['total_slots']
        best = saved['best_validation']
        torch.set_rng_state(saved['rng'])
        if saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    write_json(output / 'config.json', config)
    write_json(output / 'environment.json', snapshot)
    pool = SamplerPool(snapshot, config, cfg['workers'], saved['worker_rngs'] if saved else None)
    try:
        for episode in range(completed, cfg['episodes']):
            rows = []
            epoch = cfg['train_epoch_start'] + episode
            fraction = episode / max(1, cfg['episodes'] - 1)
            exploration = (config['high']['exploration_start'] * (1 - fraction)
                           + config['high']['exploration_end'] * fraction)
            remaining = cfg['slots']
            while remaining:
                budget = min(remaining, cfg['rollout_slots'])
                quotient, remainder = divmod(budget, cfg['workers'])
                jobs = [(index, dict(epoch=epoch, slots=quotient + int(index < remainder),
                                      evaluation=False, exploration=exploration))
                        for index in range(cfg['workers']) if quotient + int(index < remainder)]
                batch = pool.collect(jobs, low.weights(), planner.policy_state())
                low_loss = low.learn([trajectory for result in batch for trajectory in result['trajectories']])
                high_loss = planner.fit_experiences([item for result in batch for item in result['experiences']])
                for result in batch:
                    rows.extend(result['rows'])
                remaining -= budget
                total_slots += budget
                write_json(output / 'status.json', dict(status='training', episode=episode + 1,
                           episodes=cfg['episodes'], episode_slots=cfg['slots'] - remaining,
                           total_slots=total_slots, low_loss=low_loss, high_loss=high_loss))
            completed = episode + 1
            state = dict(format='mrdts_delay_v1', config=config, environment=snapshot,
                         completed_episodes=completed, total_slots=total_slots,
                         low=low.state(), high=planner.state(), best_validation=best,
                         rng=torch.get_rng_state(), worker_rngs=pool.rngs,
                         cuda_rng=torch.cuda.get_rng_state_all() if low.device.type == 'cuda' else None)
            metrics = summarize(rows)
            metrics.update(episode=completed, exploration=exploration, low_loss=low_loss, high_loss=high_loss)
            if cfg['validation_interval'] and completed % cfg['validation_interval'] == 0:
                validation = _evaluate_state(state, cfg['validation_epochs'], cfg['evaluation_slots'], cfg['workers'])
                score = validation['summary']['mean_delay_s']
                metrics['validation_mean_delay_s'] = score
                write_json(output / f'validation_{completed:04d}.json', validation)
                if score < best:
                    best = score
                    state['best_validation'] = best
                    save_checkpoint(output / 'best.pt', state)
            # Commit a complete episode before emitting its log records. Checkpoints
            # include optimizer, replay, fitting RNG and every sampler's action RNG.
            save_checkpoint(output / 'last.pt', state)
            if completed % cfg['checkpoint_interval'] == 0:
                save_checkpoint(output / f'episode_{completed:04d}.pt', state)
            append_rows(output / 'slots.csv', [dict(episode=completed, **row) for row in rows])
            # JSON lines allow validation fields to appear only on validation episodes.
            with (output / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metrics) + '\n')
            print(f"episode {completed}/{cfg['episodes']} slots={total_slots} "
                  f"mean_delay_s={metrics['mean_delay_s']:.6f}", flush=True)
        write_json(output / 'status.json', dict(status='completed', completed_episodes=completed,
                                                total_slots=total_slots, best_validation=best))
    finally:
        pool.close()
    return output / 'last.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--slots', type=int)
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.resume is not None:
        config = torch.load(args.resume, map_location='cpu', weights_only=True)['config']
        output = args.output or args.resume.parent
    else:
        for name in ('slots', 'workers'):
            value = getattr(args, name)
            if value is not None:
                config['train'][name] = value
        output = args.output or Path(config['train']['output_root']) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    train(config, output, resume=args.resume, episodes=args.episodes)


if __name__ == '__main__':
    main()
