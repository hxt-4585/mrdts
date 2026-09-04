"""Plot recorded RL metrics with raw data and an explicitly labelled rolling mean."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Plot real training logs; does not run training')
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'scripts/output/rl-training-curves')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .8, 'legend.frameon': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    figure = plt.figure(figsize=(7.1, 3.8), layout='constrained')
    grid = figure.add_gridspec(2, 2, width_ratios=[1.65, 1])
    delay, failures, validation = (figure.add_subplot(grid[:, 0]),
                                   figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, 1]))
    source = []
    palette = ['#4477AA', '#EE6677', '#228833', '#AA3377', '#66CCEE']
    for index, directory in enumerate(args.runs):
        config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        seed = config['training']['seed']
        with (directory / 'metrics.csv').open(encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        source.extend([dict(seed=seed, **row) for row in rows])
        member = [row for row in rows if row['stage'] == 'member']
        if not member:
            raise ValueError(f'No Member training rows for seed {seed}')
        steps = np.array([int(row['update']) for row in member])
        values = np.array([float(row['mean_delay_s']) for row in member])
        rates = np.array([float(row['failure_rate']) * 100 for row in member])
        if not np.isfinite(values).all() or not np.isfinite(rates).all():
            raise ValueError('Cannot plot nonfinite training metrics')
        width = min(5, len(values))
        smooth = lambda data: np.convolve(data, np.ones(width) / width, mode='valid')
        color = palette[index % len(palette)]
        delay.plot(steps, values, color=color, linewidth=.6, alpha=.2)
        delay.plot(steps[width - 1:], smooth(values), color=color, linewidth=1.4, label=f'Seed {seed}')
        failures.plot(steps[width - 1:], smooth(rates), color=color, linewidth=1.1)
        checks = [row for row in rows if row['stage'] in ('initial_member', 'member') and row['eval_delay_s']]
        validation.plot([int(row['update']) for row in checks],
                        [float(row['eval_delay_s']) for row in checks],
                        color=color, marker='o', markersize=2, linewidth=1.)
    delay.set(title='a  Sampled Member policy', xlabel='PPO update', ylabel='Censored mean DAG delay (s)')
    failures.set(title='b  Training deadline failures', ylabel='DAG failures (%)', xlabel='PPO update')
    validation.set(title='c  Deterministic validation', ylabel='Mean delay (s)', xlabel='PPO update')
    delay.legend(loc='upper right', fontsize=8)
    for axis in (delay, failures, validation):
        axis.grid(axis='y', linewidth=.5, alpha=.18)
        axis.tick_params(width=.8, length=3)
    figure.suptitle('Default environment: 100 users, 12 Member UAVs, 10 tasks per DAG', fontsize=10)
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(args.output / f'training-curves.{extension}', dpi=300, bbox_inches='tight')
    plt.close(figure)
    with (args.output / 'source-data.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, list(source[0]))
        writer.writeheader()
        writer.writerows(source)
    (args.output / 'caption.txt').write_text(
        'Raw training delay is shown faintly; bold training curves use a trailing 5-update mean '
        '(or the available length if shorter). Each update averages its complete rollout slots. '
        'Colours identify RL seeds, with identical default environment seeds and workload stream. '
        'Validation uses held-out workload slots and deterministic actions; coincident curves may '
        'overlap. Ground-biased initialization already produces an all-local deterministic policy, '
        'so its initial validation performance is not evidence of learning. No confidence intervals '
        'or theoretical convergence claim are implied. Source data contains all logged stages.', encoding='utf-8')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
