"""Train independent regional flight predictors with centralized HEFT rollouts.

Run: python -m methods.heft.run [--config PATH] [--resume last.pt]
"""
import argparse
from copy import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import tomllib

import torch

from .flight import FlightPolicy
from .rollout import make_environment, make_settings, run_slot


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'config' / 'methods' / 'heft.toml'


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open('rb') as stream:
        return tomllib.load(stream)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def summarize(rows):
    count = sum(row['dag_count'] for row in rows)
    delay = sum(row['delay_sum_s'] for row in rows)
    failures = sum(row['failed_dags'] for row in rows)
    return dict(slots=len(rows), dag_count=count, delay_sum_s=delay, mean_delay_s=delay / count,
                failed_dags=failures, failure_rate=failures / count,
                flight_violations=sum(row['flight_violations'] for row in rows),
                total_energy_j=sum(row['total_energy_j'] for row in rows),
                planning_seconds=sum(row['planning_seconds'] for row in rows))


def assess(snapshot, config, state, epochs, slots):
    # Evaluation owns separate networks, buffers and scene state.
    settings = dict(snapshot, env=dict(snapshot['env'], steps_per_epoch=slots))
    env = make_environment(settings)
    policy = FlightPolicy(env, config['high'])
    policy.restore(state)
    rows = []
    for epoch in epochs:
        env.reset(epoch_index=epoch)
        policy.reset(epoch)
        for _ in range(slots):
            row, _ = run_slot(env, policy)
            rows.append(row)
    return dict(epochs=list(epochs), summary=summarize(rows), rows=rows)


def evaluate(checkpoint, epochs=None, slots=None):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = saved['config']
    return assess(saved['environment'], config, saved['flight'],
                  config['train']['test_epochs'] if epochs is None else epochs,
                  config['train']['evaluation_slots'] if slots is None else slots)


def train(config, output, resume=None, stop_after=None, episodes=None, slots=None):
    """Train whole episodes; stop_after supports bounded resumability checks."""
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if saved is not None:
        config = saved['config']
    cfg = config['train']
    if episodes is not None:
        cfg['episodes'] = episodes
    if slots is not None:
        cfg['slots'] = slots
    snapshot = saved['environment'] if saved else make_settings(cfg)
    snapshot['env']['steps_per_epoch'] = cfg['slots']
    environments = [make_environment(snapshot) for _ in range(cfg['samplers'])]
    learner = FlightPolicy(environments[0], config['high'])
    completed, best = 0, None
    if saved:
        learner.restore(saved['flight'])
        completed, best = saved['completed_episodes'], saved['best_validation']
    # Copies share trained networks; reset creates each sampler's flight state.
    actors = [copy(learner) for _ in environments]
    write_json(output / 'config.json', config)
    write_json(output / 'environment.json', snapshot)
    end_episode = cfg['episodes'] if stop_after is None else min(cfg['episodes'], completed + stop_after)
    for episode in range(completed, end_episode):
        epoch = cfg['train_epoch_start'] + episode
        for sampler, (env, actor) in enumerate(zip(environments, actors)):
            env.reset(epoch_index=epoch)
            actor.reset(epoch, sampler=sampler)
        fraction = episode / max(cfg['episodes'] - 1, 1)
        high = config['high']
        exploration = high['exploration_start'] + fraction * (high['exploration_end'] - high['exploration_start'])
        rows, losses = [], []
        while len(rows) < cfg['slots']:
            experiences = []
            budget = min(cfg['rollout_slots'], cfg['slots'] - len(rows))
            for _ in range(budget):
                index = len(rows) % len(environments)
                actor = actors[index]
                actor.samples = learner.samples
                row, experience = run_slot(environments[index], actor, exploration)
                row['sampler'] = index
                rows.append(row)
                experiences.append(experience)
            losses.append(learner.learn(experiences))
            write_json(output / 'progress.json', dict(
                status='training', episode=episode + 1, total_episodes=cfg['episodes'],
                slots_completed=len(rows), slots_per_episode=cfg['slots'],
                completed_episodes=completed, updated_at=datetime.now().isoformat()))
        completed = episode + 1
        metrics = dict(episode=completed, exploration=exploration,
                       high_loss=sum(losses) / len(losses), **summarize(rows))
        flight_state = learner.state()
        if cfg['validation_interval'] and completed % cfg['validation_interval'] == 0:
            validation = assess(snapshot, config, flight_state, cfg['validation_epochs'], cfg['evaluation_slots'])
            write_json(output / f'validation_{completed:04d}.json', validation)
            score = validation['summary']['mean_delay_s']
            metrics['validation_mean_delay_s'] = score
            improved = best is None or score < best
            if improved:
                best = score
        else:
            improved = False
        state = dict(format='heft_learned_flight_v1', config=config, environment=snapshot,
                     flight=flight_state, completed_episodes=completed, best_validation=best)
        save_checkpoint(output / 'last.pt', state)
        if improved:
            save_checkpoint(output / 'best.pt', state)
        if completed % cfg['checkpoint_interval'] == 0:
            save_checkpoint(output / f'episode_{completed:04d}.pt', state)
        with (output / 'slots.csv').open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['episode', *rows[0]])
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerows(dict(episode=completed, **row) for row in rows)
        with (output / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + '\n')
        print(f"episode {completed}/{cfg['episodes']} mean_delay_s={metrics['mean_delay_s']:.6f} "
              f"failure_rate={metrics['failure_rate']:.4f} high_loss={metrics['high_loss']:.6f}", flush=True)
    write_json(output / 'progress.json', dict(
        status='completed' if completed == cfg['episodes'] else 'stopped',
        completed_episodes=completed, total_episodes=cfg['episodes'],
        updated_at=datetime.now().isoformat()))
    return output / 'last.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--slots', type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.episodes is not None:
        config['train']['episodes'] = args.episodes
    if args.slots is not None:
        config['train']['slots'] = args.slots
    output = args.output or (args.resume.parent if args.resume else
                             Path(config['train']['output_root']) / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    train(config, output, resume=args.resume, episodes=args.episodes, slots=args.slots)


if __name__ == '__main__':
    main()
