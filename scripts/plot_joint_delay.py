"""Plot the two joint-training runs with shared axes and inspectable source data."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_run(root, method):
    manifest_path = root / 'launch.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        directory = Path(manifest['runs'][method]['output'])
    else:
        directory = root / method
    with (directory / 'episodes.csv').open(encoding='utf-8', newline='') as stream:
        episodes = list(csv.DictReader(stream))
    train_x = np.asarray([int(row['episode']) for row in episodes])
    train_y = np.asarray([float(row['mean_delay_s']) * 1000 for row in episodes])
    smooth = np.asarray([train_y[max(0, i - 4):i + 1].mean() for i in range(len(train_y))])
    validation = []
    for path in sorted(directory.glob('validation_*.json')):
        epoch = int(path.stem.split('_')[1])
        rows = json.loads(path.read_text(encoding='utf-8'))
        for row in rows:
            validation.append(dict(episode=epoch, epoch=row['epoch'], delay_ms=row['mean_delay_s'] * 1000,
                                   dag_count=row['dag_count'], failed_dags=row['failed_dags'],
                                   flight_violations=row['flight_violations']))
    val_x = sorted({row['episode'] for row in validation})
    val_y = [np.mean([r['delay_ms'] for r in validation if r['episode'] == episode]) for episode in val_x]
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))['config']
    return dict(method=method, episodes=episodes, train_x=train_x, train_y=train_y, smooth=smooth,
                val_x=np.asarray(val_x), val_y=np.asarray(val_y), validation=validation,
                metadata=metadata, config=config)


def plot_combined(root, *, reward=False):
    root = Path(root).resolve()
    output = root / 'figures'
    output.mkdir(exist_ok=True)
    datasets = [read_run(root, method) for method in ('joint_delay_ppo', 'joint_delay_acac')]
    if reward:
        manifest = json.loads((root / 'launch.json').read_text(encoding='utf-8-sig'))
        for data in datasets:
            run = Path(manifest['runs'][data['method']]['output'])
            config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
            duration = config['environment']['env']['scheduling_duration_s']
            data['train_y'] = -data['train_y'] / (1000 * duration)
            data['smooth'] = -data['smooth'] / (1000 * duration)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'legend.frameon': False})
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    fig.subplots_adjust(left=.12, right=.97, top=.87, bottom=.22)
    for data, name, color, style in zip(datasets, ('PPO', 'ACAC'), ('#356A99', '#C67843'), ('-', '--')):
        ax.plot(data['train_x'], data['train_y'], color=color, alpha=.25, lw=.85, label=f'{name}: raw')
        ax.plot(data['train_x'], data['smooth'], color=color, lw=1.9, ls=style, label=f'{name}: 5-ave')
    values = np.concatenate([data['train_y'] for data in datasets])
    low = .05 * np.floor((values.min() - .015) / .05) if reward else 20 * np.floor((values.min() - 15) / 20)
    high = .05 * np.ceil((values.max() + .015) / .05) if reward else 20 * np.ceil((values.max() + 15) / 20)
    planned = max(data['config']['train']['episodes'] for data in datasets)
    ax.set(title='Joint training: PPO vs ACAC', xlabel='Training episode',
           ylabel='Mean system reward per slot' if reward else 'Mean DAG completion delay (ms)',
           xlim=(0, planned + 1), ylim=(low, high))
    ax.set_xticks(np.arange(0, planned + 1, 20))
    spacing = .05 if reward else 40
    ax.set_yticks(np.arange(np.ceil(low / spacing) * spacing, high + spacing * .01, spacing))
    ax.grid(axis='y', color='#E7EAED', lw=.6)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right' if reward else 'upper right', ncol=2, fontsize=8.5, columnspacing=1.8)
    note = ('System reward = -mean DAG delay / scheduling duration; not the Member PPO return.' if reward
            else '5-ave: trailing mean of 5 episodes; first 4 points use the available preceding episodes.')
    fig.text(.12, .09, note,
             fontsize=7.7, color='#687581')
    fig.text(.12, .035,
             f'PPO: {len(datasets[0]["train_x"])} episodes; ACAC: {len(datasets[1]["train_x"])} episodes. '
             f'Trailing 5-episode means; seed 45. {"Higher" if reward else "Lower"} is better.', fontsize=7.7, color='#687581')
    base = output / ('ppo_acac_system_reward' if reward else 'ppo_acac_training_delay')
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(base.with_suffix('.' + extension), dpi=300, facecolor='white')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_text() and text.get_visible():
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                clipped.append(text.get_text())
    line_count = len(ax.lines)
    plt.close(fig)
    records = {}
    for data, name in zip(datasets, ('ppo', 'acac')):
        for epoch, raw, smooth in zip(data['train_x'], data['train_y'], data['smooth']):
            row = records.setdefault(int(epoch), dict(episode=int(epoch)))
            row.update({f'{name}_raw': float(raw), f'{name}_5ave': float(smooth)})
    with base.with_suffix('.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['episode', 'ppo_raw', 'ppo_5ave', 'acac_raw', 'acac_5ave'])
        writer.writeheader()
        writer.writerows(records[epoch] for epoch in sorted(records))
    qa = dict(curves=line_count, ppo_episodes=len(datasets[0]['train_x']),
              acac_episodes=len(datasets[1]['train_x']), text_outside_canvas=clipped,
              metric='mean system reward per slot' if reward else 'mean DAG delay in ms',
              source='completed training episodes; no validation or extrapolation',
              smoothing='trailing five episodes, shorter window for the first four points')
    base.with_suffix('.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa, indent=2))


def plot(root):
    root = Path(root).resolve()
    output = root / 'figures'
    output.mkdir(exist_ok=True)
    datasets = [read_run(root, method) for method in ('joint_delay_ppo', 'joint_delay_acac')]
    all_values = np.concatenate([np.concatenate((data['train_y'], data['val_y'])) for data in datasets])
    lower = 20 * np.floor((all_values.min() - 15) / 20)
    upper = 20 * np.ceil((all_values.max() + 15) / 20)
    planned = max(data['config']['train']['episodes'] for data in datasets)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .7, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'legend.frameon': False, 'savefig.facecolor': 'white'})
    records, qa = [], []
    titles = {'joint_delay_ppo': 'Joint training: independent PPO',
              'joint_delay_acac': 'Joint training: ACAC'}
    for data in datasets:
        fig, ax = plt.subplots(figsize=(7.2, 4.25))
        fig.subplots_adjust(left=.12, right=.97, top=.87, bottom=.23)
        ax.set_title(titles[data['method']], loc='left', pad=13, weight='bold')
        ax.plot(data['train_x'], data['train_y'], color='#9BA7B3', alpha=.55, lw=.8,
                label='Training: individual episodes')
        ax.plot(data['train_x'], data['smooth'], color='#356A99', lw=1.8,
                label='Training: trailing 5-episode mean')
        ax.plot(data['val_x'], data['val_y'], color='#C67843', lw=1.6, marker='o', ms=3.5,
                label='Validation: mean of 3 task streams')
        best_index = int(np.argmin(data['val_y']))
        best_episode, best_delay = data['val_x'][best_index], data['val_y'][best_index]
        ax.scatter([best_episode], [best_delay], marker='*', s=105, color='#C67843',
                   edgecolors='white', linewidths=.6, zorder=6)
        ax.annotate(f'Best validation: {best_delay:.2f} ms\nEpisode {best_episode}',
                    xy=(best_episode, best_delay), xytext=(best_episode - 22, lower + .36 * (upper - lower)),
                    fontsize=8.5, color='#80502F', ha='left', va='center',
                    arrowprops=dict(arrowstyle='-', color='#C67843', lw=.8),
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2))
        completed = data['metadata']['episodes_completed']
        if data['metadata']['status'] != 'completed':
            ax.axvspan(completed, planned, facecolor='#E8EBEE', alpha=.55, zorder=0)
            ax.axvline(completed, ymax=.74, color='#73808C', ls=(0, (3, 3)), lw=.9)
            ongoing = data['metadata']['status'] == 'running'
            ax.text((completed + planned) / 2, lower + .73 * (upper - lower),
                    'Training\nongoing' if ongoing else 'Not\ncompleted',
                    color='#687581', fontsize=8.5, ha='center', va='center')
            status = (f'Training ongoing: {completed} complete episodes; planned: {planned}.' if ongoing
                      else f'Interrupted after {completed} complete episodes; planned: {planned}.')
        else:
            status = f'Completed all {completed} episodes.'
        ax.set(xlim=(0, planned + 1), ylim=(lower, upper), xlabel='Training episode',
               ylabel='Mean DAG completion delay (ms)')
        ax.set_xticks(np.arange(0, planned + 1, 20))
        ax.set_yticks(np.arange(np.ceil(lower / 40) * 40, upper + 1, 40))
        ax.grid(axis='y', color='#E7EAED', lw=.6)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', fontsize=8, handlelength=2.2, labelspacing=.45)
        fig.text(.12, .085, status, fontsize=8.5, color='#49545F')
        fig.text(.12, .035, 'Single training seed (45). Each validation stream contains 500 slots / 10,000 DAGs.',
                 fontsize=7.5, color='#687581')
        base = output / data['method']
        for extension in ('svg', 'pdf', 'png'):
            fig.savefig(base.with_suffix('.' + extension), dpi=300)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [text.get_window_extent(renderer) for text in fig.findobj(matplotlib.text.Text)
                 if text.get_text() and text.get_visible()]
        outside = [box for box in boxes if box.x0 < -1 or box.y0 < -1 or
                   box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1]
        qa.append(dict(method=data['method'], training_episodes=len(data['train_x']),
                       validation_checkpoints=len(data['val_x']),
                       best_episode=int(best_episode), best_validation_ms=float(best_delay),
                       metadata_selected_episode=data['metadata']['selected_episode'],
                       text_outside_canvas=len(outside), x_limits=list(ax.get_xlim()),
                       y_limits=list(ax.get_ylim()), smoothing='trailing mean, window 5; shorter window at start',
                       statistics='validation means over task streams, not independent training seeds'))
        plt.close(fig)
        for source, episode, raw, smoothed in zip(data['episodes'], data['train_x'], data['train_y'], data['smooth']):
            records.append(dict(method=data['method'], split='train', episode=int(episode), epoch=int(source['epoch']),
                                delay_ms=float(raw), trailing_mean_ms=float(smoothed), dag_count=int(source['dag_count']),
                                failed_dags=int(source['failed_dags']), flight_violations=int(source['flight_violations'])))
        for row in data['validation']:
            records.append(dict(method=data['method'], split='validation', trailing_mean_ms='', **row))
    with (output / 'source_data.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / 'figure_qa.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa, indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--combined', action='store_true')
    parser.add_argument('--reward', action='store_true', help='Compare recoverable system rewards instead of delay')
    args = parser.parse_args()
    if args.reward:
        plot_combined(args.run_root, reward=True)
    else:
        (plot_combined if args.combined else plot)(args.run_root)
