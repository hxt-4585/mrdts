"""Plot recoverable system reward; historical Member PPO returns were not logged."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(run):
    run = Path(run).resolve()
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    duration = config['environment']['env']['scheduling_duration_s']
    with (run / 'episodes.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    episode = np.asarray([int(row['episode']) for row in rows])
    reward = -np.asarray([float(row['mean_delay_s']) for row in rows]) / duration
    moving = np.asarray([reward[max(0, i - 4):i + 1].mean() for i in range(len(reward))])
    planned = config['config']['train']['episodes']
    output = run.parent / 'figures'
    output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .7, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'legend.frameon': False})
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    fig.subplots_adjust(left=.14, right=.97, top=.86, bottom=.24)
    ax.set_title('System reward during ACAC joint training', loc='left', weight='bold', pad=14)
    ax.plot(episode, reward, color='#9AA9B7', lw=.85, alpha=.55, label='Individual episodes')
    ax.plot(episode, moving, color='#356A99', lw=1.9, label='Trailing 5-episode mean')
    lower = .05 * np.floor((reward.min() - .015) / .05)
    upper = .05 * np.ceil((reward.max() + .015) / .05)
    if episode[-1] < planned:
        ax.axvspan(episode[-1], planned, facecolor='#E8EBEE', alpha=.5, zorder=0)
        ax.text((episode[-1] + planned) / 2, lower + .3 * (upper - lower),
                'Training\nongoing', ha='center', va='center', color='#687581', fontsize=8)
    for boundary in (106, 110):
        if boundary < episode[-1]:
            ax.axvline(boundary, color='#A1ABB4', lw=.75, ls=(0, (3, 3)), ymax=.66)
    ax.set(xlim=(0, planned + 1), ylim=(lower, upper), xlabel='Training episode',
           ylabel='Mean system reward per slot')
    ax.set_xticks(np.arange(0, planned + 1, 20))
    ax.set_yticks(np.arange(np.ceil(lower / .05) * .05, upper + .001, .05))
    ax.grid(axis='y', color='#E7EAED', lw=.6)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=8.5)
    fig.text(.14, .115, f'Reward = -mean DAG delay / {duration:g} s; higher is better. '
             f'{episode[-1]} complete episodes shown.', color='#49545F', fontsize=8)
    fig.text(.14, .065, 'System-level metric, not the Member PPO training return. Single training seed (45).',
             color='#687581', fontsize=7.8)
    fig.text(.14, .02, 'Dashed lines: checkpoint resumptions after episodes 106 and 110. No extrapolated rewards.',
             color='#687581', fontsize=7.8)
    base = output / 'acac_system_reward'
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(base.with_suffix('.' + extension), dpi=300, facecolor='white')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_text() and text.get_visible():
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                outside.append(text.get_text())
    plt.close(fig)
    with base.with_suffix('.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['episode', 'epoch', 'mean_delay_s', 'system_reward', 'trailing_mean_5'])
        writer.writeheader()
        for row, value, smoothed in zip(rows, reward, moving):
            writer.writerow(dict(episode=row['episode'], epoch=row['epoch'], mean_delay_s=row['mean_delay_s'],
                                 system_reward=value, trailing_mean_5=smoothed))
    qa = dict(generated_at=datetime.now(timezone.utc).isoformat(), complete_episodes=int(episode[-1]),
              metric='negative mean DAG delay divided by nominal scheduling duration',
              is_member_training_return=False, initial_episode_reward=float(reward[0]),
              latest_episode_reward=float(reward[-1]), latest_trailing_mean=float(moving[-1]),
              mean_first_10=float(reward[:10].mean()), mean_last_10=float(reward[-10:].mean()),
              text_outside_canvas=outside, source=str(run / 'episodes.csv'))
    base.with_suffix('.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    plot(parser.parse_args().run)
