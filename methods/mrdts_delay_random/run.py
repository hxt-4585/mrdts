"""Train the regional delay predictor using random offloading rollouts."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import tomllib

import torch

from .high import RegionalDelays
from .rollout import Sampler, environment_snapshot


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'config/methods/mrdts_delay_random.toml'


def load_config(path=DEFAULT_CONFIG):
    with path.open('rb') as stream:
        return tomllib.load(stream)


def predictor(snapshot, config):
    return RegionalDelays(snapshot['uav']['member_uav_count'], snapshot['region']['region_count'],
                          config['high'])


def summary(rows):
    count = sum(row['dag_count'] for row in rows)
    delay = sum(row['delay_sum_s'] for row in rows)
    failed = sum(row['failed_dags'] for row in rows)
    return dict(slots=len(rows), dag_count=count, delay_sum_s=delay, mean_delay_s=delay / count,
                failed_dags=failed, failure_rate=failed / count,
                flight_violations=sum(row['flight_violations'] for row in rows))


def assess(snapshot, config, model, epochs, slots):
    sampler = Sampler(snapshot, config['train']['seed'], 0)
    rows = []
    for epoch in epochs:
        sampler.reset(epoch, model)
        for _ in range(slots):
            row, _ = sampler.step(0., training=False)
            rows.append(row)
    return dict(epochs=list(epochs), summary=summary(rows), rows=rows)


def evaluate(checkpoint, epochs=None, slots=None):
    torch.set_num_threads(1)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config, snapshot = state['config'], state['environment']
    model = predictor(snapshot, config)
    model.restore(state['high'], training=False)
    return assess(snapshot, config, model, epochs if epochs is not None else config['train']['test_epochs'],
                  slots if slots is not None else config['train']['evaluation_slots'])


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def save(path, state):
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def train(config, output, resume=None, episodes=None):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None:
        snapshot = environment_snapshot(config['train'])
        completed, best = 0, None
        model = predictor(snapshot, config)
    else:
        state = torch.load(resume, map_location='cpu', weights_only=True)
        config, snapshot = state['config'], state['environment']
        completed, best = state['completed_episodes'], state['best_validation']
        model = predictor(snapshot, config)
        model.restore(state['high'], training=True)
    cfg = config['train']
    if episodes is not None:
        cfg['episodes'] = episodes
    write_json(output / 'config.json', config)
    write_json(output / 'environment.json', snapshot)
    samplers = [Sampler(snapshot, cfg['seed'], i) for i in range(cfg['samplers'])]
    for episode in range(completed, cfg['episodes']):
        epoch = cfg['train_epoch_start'] + episode
        for sampler in samplers:
            sampler.reset(epoch, model)
        fraction = episode / max(cfg['episodes'] - 1, 1)
        high = config['high']
        exploration = (1 - fraction) * high['exploration_start'] + fraction * high['exploration_end']
        rows = []
        while len(rows) < cfg['slots']:
            budget = min(cfg['slots'] - len(rows), cfg['rollout_slots'])
            per_sampler, extra = divmod(budget, len(samplers))
            experiences = []
            for i, sampler in enumerate(samplers):
                for _ in range(per_sampler + int(i < extra)):
                    row, experience = sampler.step(exploration, training=True)
                    rows.append(row)
                    experiences.append(experience)
            loss = model.update(experiences)
        completed = episode + 1
        metrics = dict(episode=completed, exploration=exploration, high_loss=loss, **summary(rows))
        state = dict(format='mrdts_delay_random_v1', environment=snapshot, config=config,
                     high=model.checkpoint(), completed_episodes=completed, best_validation=best)
        if cfg['validation_interval'] and completed % cfg['validation_interval'] == 0:
            report = assess(snapshot, config, model, cfg['validation_epochs'], cfg['evaluation_slots'])
            write_json(output / f'validation_{completed:04d}.json', report)
            score = report['summary']['mean_delay_s']
            metrics['validation_mean_delay_s'] = score
            if best is None or score < best:
                best = state['best_validation'] = score
                save(output / 'best.pt', state)
        save(output / 'last.pt', state)
        if completed % cfg['checkpoint_interval'] == 0:
            save(output / f'episode_{completed:04d}.pt', state)
        path = output / 'slots.csv'
        with path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, ['episode', *rows[0]])
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerows(dict(episode=completed, **row) for row in rows)
        with (output / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics) + '\n')
        print(f"episode {completed}/{cfg['episodes']} mean_delay_s={metrics['mean_delay_s']:.6f}", flush=True)
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
    if args.slots is not None:
        config['train']['slots'] = args.slots
    output = args.output
    if output is None:
        output = args.resume.parent if args.resume else (
            Path(config['train']['output_root']) / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    train(config, output, args.resume, args.episodes)


if __name__ == '__main__':
    main()
