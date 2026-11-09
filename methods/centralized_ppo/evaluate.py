"""Re-evaluate the selected neural actor and reference policies on paired tasks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from methods.centralized_ppo.run import final_evaluation
from methods.centralized_ppo.simulation import initialize_worker


def evaluate(run, epoch_start, episodes, workers):
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    config['test_epochs'] = list(range(epoch_start, epoch_start + episodes))
    report = run / f'eval_{epoch_start}_{epoch_start + episodes - 1}'
    report.mkdir()
    with ProcessPoolExecutor(workers, initializer=initialize_worker,
                             initargs=(run / 'frozen_members.pt',)) as pool:
        final_evaluation(pool, run, config, report)
    return report / 'test_results.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--epoch-start', type=int, default=2100000)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    print(evaluate(**vars(parser.parse_args())))
