"""Frozen evaluation and episode-level paired noninferiority summaries."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from methods.distributed_delay.planner import DistributedController, frames_from_env
from methods.distributed_delay.simulator import FrozenSimulation


def initialize(source_path, config):
    global SIMULATION, CONFIG
    torch.set_num_threads(1)
    SIMULATION = FrozenSimulation(torch.load(source_path, map_location='cpu', weights_only=True))
    CONFIG = config


def evaluate_one(job):
    snapshot, epoch, slots, targets = job
    if targets is not None:
        return SIMULATION.episode(epoch, slots, targets=targets)
    controller = DistributedController(frames_from_env(SIMULATION.env), CONFIG)
    saved = torch.load(snapshot, map_location='cpu', weights_only=True)
    for model, state in zip(controller.models, saved['models']):
        model.load_state_dict(state['model'])
        model.samples = state['samples']
    return SIMULATION.episode(epoch, slots, controller=controller)


class Evaluation:
    def __init__(self, source_path, config):
        self.config = config
        self.pool = None
        if config['workers'] == 1:
            initialize(source_path, config)
        else:
            self.pool = ProcessPoolExecutor(max_workers=config['workers'], initializer=initialize,
                                            initargs=(source_path, config))

    def episodes(self, snapshot, epochs, *, targets=None):
        jobs = [(snapshot, epoch, self.config['evaluation_slots'], targets) for epoch in epochs]
        random_state = torch.get_rng_state()
        try:
            return list(map(evaluate_one, jobs) if self.pool is None else self.pool.map(evaluate_one, jobs))
        finally:
            torch.set_rng_state(random_state)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


def compare(learned, reference, *, seed):
    a = np.array([row['mean_delay_s'] for row in learned])
    b = np.array([row['mean_delay_s'] for row in reference])
    bootstrap_seed = int(np.random.SeedSequence([seed, 103]).generate_state(1)[0])
    indices = np.random.default_rng(bootstrap_seed).integers(len(a), size=(10000, len(a)))
    differences = a - b
    margin = a - 1.01 * b
    interval = np.quantile(margin[indices].mean(1), [.025, .975])
    return dict(bootstrap_seed=bootstrap_seed,
                learned_delay_s=float(a.mean()), reference_delay_s=float(b.mean()),
                relative_delay_difference=float(a.mean() / b.mean() - 1.),
                paired_delay_difference_95_ci_s=np.quantile(differences[indices].mean(1), [.025, .975]).tolist(),
                paired_1pct_margin_95_ci_s=interval.tolist(),
                point_within_1pct=bool(a.mean() <= 1.01 * b.mean()),
                ci_supports_1pct=bool(interval[1] <= 0.),
                failure_rate_difference=float(np.mean([row['failure_rate'] for row in learned]) -
                                              np.mean([row['failure_rate'] for row in reference])))


def final_comparison(run, config, evaluator):
    # This is the sole production read of CEM's selected deployment.
    reference = torch.load(config['cem_reference'], map_location='cpu', weights_only=True)
    targets = reference['targets'].numpy()
    selected = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    learned = evaluator.episodes(run / 'best.pt', config['test_epochs'])
    cem = evaluator.episodes(None, config['test_epochs'], targets=targets)
    summary = compare(learned, cem, seed=config['seed'])
    return dict(episodes=dict(distributed_delay=learned, cem=cem), comparison=summary,
                selected_episode=selected['episode'],
                gradient_planner_active=all(state['samples'] >= config['warmup_samples']
                                            for state in selected['models']),
                information='learned: own + one-hop; CEM: centralized static deployment',
                criterion='paired episode bootstrap; 1% mean-delay noninferiority tolerance',
                reference_member_source=reference.get('member_source'),
                matches_cem=summary['ci_supports_1pct'] and summary['failure_rate_difference'] <= 0.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    config = json.loads((args.run / 'config.json').read_text(encoding='utf-8'))['training']
    evaluator = Evaluation(args.run / 'frozen_members.pt', config)
    try:
        result = final_comparison(args.run, config, evaluator)
        (args.run / 'evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result['comparison']), flush=True)
    finally:
        evaluator.close()
