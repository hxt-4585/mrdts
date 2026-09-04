"""Source-backed mean-reward and epoch-return plots for new or legacy logs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rewards(directory):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    legacy = not (directory / 'epochs.csv').exists()
    path = directory / ('metrics.csv' if legacy else 'epochs.csv')
    with path.open(newline='', encoding='utf-8') as file:
        rows = [row for row in csv.DictReader(file) if row['stage'] in ('member', 'master')]
    if not rows:
        raise ValueError('No completed training epochs/rounds have been logged yet')
    steps = ([int(config['training']['rollout_slots'])] * len(rows) if legacy
             else [int(row['steps']) for row in rows])
    means = [float(row['reward' if legacy else 'mean_reward']) for row in rows]
    returns = ([value * count for value, count in zip(means, steps)] if legacy
               else [float(row['epoch_return']) for row in rows])
    if not np.isfinite(means + returns).all() or not np.allclose(returns, np.array(means) * steps):
        raise ValueError('Reward totals and means are inconsistent or nonfinite')
    return dict(seed=config['training']['seed'], legacy=legacy, steps=steps, means=means,
                returns=returns, epochs=list(range(1, len(rows) + 1)) if legacy
                else [int(row['epoch']) for row in rows], stages=[row['stage'] for row in rows])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Plot mean reward and total reward per full environment epoch')
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    data = [read_rewards(path) for path in args.runs]
    if len({record['legacy'] for record in data}) != 1:
        raise ValueError('Plot legacy rounds and full-step epochs separately')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.spines.right': False, 'axes.spines.top': False,
                         'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout='constrained')
    colours = ['#4477AA', '#EE6677', '#228833', '#AA3377']
    source = []
    for index, record in enumerate(data):
        x = np.array(record['epochs'])
        colour = colours[index % len(colours)]
        for axis, field in zip(axes, ('means', 'returns')):
            values = np.array(record[field])
            axis.plot(x, values, color=colour, alpha=.3, linewidth=.8)
            width = min(5, len(x))
            # Preserve every point for short smoke experiments; no invented history.
            if len(x) < 5:
                axis.plot(x, values, marker='o', color=colour, label=f"Seed {record['seed']}")
            else:
                axis.plot(x[width - 1:], np.convolve(values, np.ones(width) / width, mode='valid'),
                          color=colour, linewidth=1.6, label=f"Seed {record['seed']}")
        source.extend(dict(seed=record['seed'], legacy=record['legacy'], epoch=epoch, stage=stage,
                           steps=steps, mean_reward=mean, epoch_return=total)
                      for epoch, stage, steps, mean, total in zip(
                          record['epochs'], record['stages'], record['steps'], record['means'], record['returns']))
    legacy = data[0]['legacy']
    counts = sorted({count for record in data for count in record['steps']})
    unit = 'collection round' if legacy else 'epoch'
    count_label = '/'.join(map(str, counts))
    figure.suptitle(f"{'Legacy' if legacy else 'Full-step'} reward | {count_label} environment steps per {unit}")
    boundary = next((epoch for epoch, stage in zip(data[0]['epochs'], data[0]['stages']) if stage == 'master'), None)
    for axis in axes:
        axis.set_xlabel('Collection round (legacy)' if legacy else 'Epoch')
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(axis='y', linewidth=.5, alpha=.2)
        if boundary is not None:
            axis.axvline(boundary - .5, color='#888888', linestyle='--', linewidth=.8)
    axes[0].set_title('Mean step reward (higher is better)')
    axes[0].set_ylabel('Mean reward')
    axes[1].set_title('Sum of step rewards')
    axes[1].set_ylabel('Round return' if legacy else 'Epoch return')
    axes[0].legend()
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(args.output / f'epoch-reward.{extension}', dpi=250, bbox_inches='tight')
    plt.close(figure)
    with (args.output / 'reward-source.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, list(source[0]))
        writer.writeheader()
        writer.writerows(source)
    caption = ('Reward is the negative normalized global mean censored DAG delay. Left: mean step reward; '
               'right: sum across the reported steps. Faint lines show raw values; bold lines use a '
               'trailing five-point mean when at least five records exist. With fewer records all raw '
               'points are shown. A dashed line marks the first run\'s transition to Master updates. '
               'Source CSV records every run\'s actual stage and step count. No convergence claim is implied. ')
    if legacy:
        caption += (f'These are pre-existing {count_label}-step collection-round experiments, not 500-step epoch results. '
                    'Their totals are reconstructed from the logged mean reward and saved rollout length.')
    else:
        caption += 'Every logged environment step consists of a Master command and Member DAG scheduling.'
    (args.output / 'caption.txt').write_text(caption, encoding='utf-8')
    print(args.output.resolve())
