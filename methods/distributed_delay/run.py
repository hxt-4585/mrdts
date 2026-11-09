"""Train local delay regressors and optimize one-hop flight actions each slot.

python -B -m methods.distributed_delay.run
Only environment modules and this directory are imported by the baseline.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
from time import perf_counter
import tomllib

import numpy as np
import torch

from methods.distributed_delay.evaluate import Evaluation, final_comparison
from methods.distributed_delay.planner import DistributedController, frames_from_env
from methods.distributed_delay.simulator import FrozenSimulation


ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def train(config, *, output=None, resume=None):
    config = dict(config)
    for name in ('member_checkpoint', 'cem_reference'):
        path = Path(config[name])
        config[name] = str(path if path.is_absolute() else ROOT / path)
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    source_path = Path(config['member_checkpoint']) if resume is None else Path(resume) / 'frozen_members.pt'
    raw = source_path.read_bytes()
    source = torch.load(io.BytesIO(raw), weights_only=True, map_location='cpu')
    simulation = FrozenSimulation(source)
    controller = DistributedController(frames_from_env(simulation.env), config)
    if output is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        output = ROOT / config['output_root'] / f'seed_{config["seed"]}' / stamp
    output = Path(output).resolve()
    output.mkdir(parents=True)
    (output / 'frozen_members.pt').write_bytes(raw)
    write_json(output / 'config.json', dict(method='distributed_delay', training=config,
        environment=source['environment'], member_source_sha256=hashlib.sha256(raw).hexdigest(),
        objective='regional DAG mean delay, weighted by local user count in coordination; no energy term',
        architecture='independent local SiLU regressors; synchronous projected gradient; frozen ERS + argmax actors',
        information='own-region users; own/one-hop UAV XY and ownership; edge action/partial-gradient messages',
        proxy_target='regional total realized DAG delay / local user count',
        startup='local Lloyd coverage seeds; exploratory destinations use own-region users only',
        limitations='one-step approximate objective; unobserved distant shared-bandwidth/BS effects; fixed rounds, no convergence claim',
        selection='global validation metrics used offline only; deployment decisions remain one-hop',
        evaluation='fresh paired task streams, full travel included; no online fitting during held-out evaluation'))
    metadata = dict(status='running', pid=os.getpid(), episode=0, phase='initialization',
                    started_at=datetime.now(timezone.utc).isoformat(), elapsed_s=0.)
    started = perf_counter()
    best = float('inf')
    beginning = 0
    selected_episode = 0

    def status(**changes):
        metadata.update(changes, elapsed_s=perf_counter() - started,
                        updated_at=datetime.now(timezone.utc).isoformat())
        write_json(output / 'metadata.json', metadata)

    def save(name, episode):
        checkpoint = dict(episode=episode, best_validation_delay_s=best,
                          selected_episode=selected_episode, torch_rng=torch.get_rng_state(),
                          models=[model.training_state() for model in controller.models])
        temporary = output / (name + '.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(output / name)

    if resume is not None:
        saved = torch.load(Path(resume) / 'latest.pt', weights_only=True, map_location='cpu')
        for model, state in zip(controller.models, saved['models']):
            model.restore(state)
        beginning, best = saved['episode'], saved['best_validation_delay_s']
        selected_episode = saved['selected_episode']
        torch.set_rng_state(saved['torch_rng'])
        (output / 'best.pt').write_bytes((Path(resume) / 'best.pt').read_bytes())
    evaluator = Evaluation(output / 'frozen_members.pt', config)
    status(episode=beginning)

    def validate(episode):
        nonlocal best, selected_episode
        status(phase='validation', episode=episode)
        save('candidate.pt', episode)
        rows = evaluator.episodes(output / 'candidate.pt', config['validation_epochs'])
        delay = float(np.mean([row['mean_delay_s'] for row in rows]))
        write_json(output / f'validation_{episode:04d}.json', rows)
        if delay < best:
            best, selected_episode = delay, episode
            save('best.pt', episode)
        status(best_validation_delay_s=best, selected_episode=selected_episode)
        return delay

    try:
        if beginning == 0:
            validate(0)
            save('latest.pt', 0)
        with (output / 'episodes.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = None
            for episode in range(beginning + 1, config['episodes'] + 1):
                status(phase='training', episode=episode)
                # Absolute episode schedule preserves training under budget extensions.
                fraction = min(1., (episode - 1) / 40.)
                exploration = config['exploration_start'] * (1. - fraction) + config['exploration_end'] * fraction
                row = simulation.episode(config['train_epoch_start'] + episode - 1, config['train_slots'],
                                         controller=controller, learn=True, exploration=exploration)
                validation = None
                if episode == 1 or episode % config['validation_interval'] == 0:
                    validation = validate(episode)
                row.update(episode=episode, exploration=exploration, validation_delay_s=validation,
                           best_validation_delay_s=best, elapsed_s=perf_counter() - started)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                save('latest.pt', episode)
                status(phase='training', episode=episode, best_validation_delay_s=best,
                       planning_mean_s=row['planning_mean_s'], mean_delay_s=row['mean_delay_s'])
                print(json.dumps(row), flush=True)
        if config['final_test']:
            status(phase='final_test')
            result = final_comparison(output, config, evaluator)
            write_json(output / 'test_results.json', result)
            status(comparison=result['comparison'], matches_cem=result['matches_cem'])
        status(status='completed', phase='done', finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        evaluator.close()
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/methods/distributed_delay.toml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--skip-test', action='store_true')
    args = parser.parse_args()
    with args.config.open('rb') as stream:
        config = tomllib.load(stream)['train']
    if args.episodes is not None:
        config['episodes'] = args.episodes
    if args.skip_test:
        config['final_test'] = False
    result = train(config, output=args.output_dir, resume=args.resume_from)
    print(f'Run completed: {result}', flush=True)
