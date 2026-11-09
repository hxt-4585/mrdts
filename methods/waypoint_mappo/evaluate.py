"""Evaluate saved static deployments on paired, full-horizon workloads."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import torch

from methods.centralized_cem.run import initialize_worker
from methods.hmappo.support import write_json
from methods.waypoint_mappo.run import compare_deployments


def evaluate(run, epoch_start, episodes, workers):
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    initial, learned = [torch.load(run / name, map_location='cpu', weights_only=True)['targets'].numpy()
                        for name in ('initial.pt', 'checkpoint.pt')]
    epochs = range(epoch_start, epoch_start + episodes)
    with ProcessPoolExecutor(workers, initializer=initialize_worker,
                             initargs=(run / 'frozen_members.pt',)) as pool:
        rows = compare_deployments(pool, initial, learned, epochs, config['evaluation_slots'])
    output = run / f'eval_{epochs.start}_{epochs.stop - 1}.json'
    write_json(output, rows)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--epoch-start', type=int, default=2000000)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    print(evaluate(**vars(parser.parse_args())))
