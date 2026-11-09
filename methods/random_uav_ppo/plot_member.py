"""生成每个 Member 的完整 episode 曲线；不把缺失诊断当作零。

python methods/random_uav_ppo/plot_member.py RUN
均值按有效样本数加权；EV 使用更新中位数。return 是 GAE 目标，不是 episode 总回报。
浅色为单轮，深色为最近 10 轮。仅使用 episodes.csv 已完成的轮次。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FIELDS = dict(reward='reward_mean', return_='return_mean',
              entropy='categorical_entropy_normalized', kl='final_kl',
              clip='final_clip_fraction', actor_grad='actor_grad_norm',
              critic_grad='critic_grad_norm')


def aggregate(run, member_count):
    run = Path(run)
    with (run / 'episodes.csv').open(encoding='utf-8') as stream:
        completed = {int(row['episode']) for row in csv.DictReader(stream)}
    records = {}
    for episode in sorted(completed):
        for owner in range(member_count):
            fields = {('return' if key == 'return_' else key): value for key, value in FIELDS.items()}
            fields['self'] = f'action_{owner}_fraction'
            records[episode, owner] = dict(samples=0, fields=fields,
                sums={name: 0. for name in fields}, weights={name: 0 for name in fields}, ev=[])
    with (run / 'updates.csv').open(encoding='utf-8') as stream:
        for update in csv.DictReader(stream):
            episode = int(update['episode'])
            if episode not in completed:
                continue
            for owner in range(member_count):
                record = records[episode, owner]
                prefix = f'member_{owner}_'
                count = int(update[prefix + 'samples'])
                record['samples'] += count
                if not count:
                    continue
                for name, field in record['fields'].items():
                    value = update.get(prefix + field)
                    if value not in (None, ''):
                        record['sums'][name] += count * float(value)
                        record['weights'][name] += count
                value = update.get(prefix + 'explained_variance_after')
                if value not in (None, ''):
                    record['ev'].append(float(value))
    rows = []
    for (episode, owner), record in records.items():
        row = dict(episode=episode + 1, member=owner, samples=record['samples'])
        for field, total in record['sums'].items():
            weight = record['weights'][field]
            row[field] = total / weight if weight else None
            row[field + '_weight'] = weight
        row['ev'] = float(np.median(record['ev'])) if record['ev'] else None
        rows.append(row)
    return rows


def plot(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run = Path(run)
    with (run / 'updates.csv').open(encoding='utf-8') as stream:
        columns = next(csv.reader(stream))
    member_count = sum(name.startswith('member_') and name.endswith('_samples') and
                       name.split('_')[2] == 'samples' for name in columns)
    rows = aggregate(run, member_count)
    output = run / 'curves'
    output.mkdir(exist_ok=True)
    with (output / 'members.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = dict(reward='Reward / decision', return_='GAE target / decision',
                  entropy='Normalized entropy', self='Own execution fraction',
                  samples='Samples / episode', ev='Critic explained variance')
    for key, label in labels.items():
        field = 'return' if key == 'return_' else key
        fig, axes = plt.subplots((member_count + 3) // 4, 4, figsize=(16, 9), squeeze=False)
        for owner, axis in enumerate(axes.flat):
            if owner >= member_count:
                axis.set_visible(False)
                continue
            data = [row for row in rows if row['member'] == owner]
            x = [row['episode'] for row in data]
            y = [row[field] if row[field] is not None else np.nan for row in data]
            smoothed = []
            for index in range(len(data)):
                window = [row for row in data[max(0, index - 9):index + 1] if row[field] is not None]
                if not window:
                    smoothed.append(np.nan)
                elif field == 'ev':
                    smoothed.append(np.median([row[field] for row in window]))
                elif field == 'samples':
                    smoothed.append(np.mean([row[field] for row in window]))
                else:
                    smoothed.append(sum(row[field] * row[field + '_weight'] for row in window) /
                                    sum(row[field + '_weight'] for row in window))
            axis.plot(x, y, alpha=.3, color='#2878b5', linewidth=1, marker='.' if len(x) < 5 else None)
            axis.plot(x, smoothed, color='#164f78', linewidth=1.5)
            if all(value is None for value in (row[field] for row in data)):
                message = 'No samples' if not sum(row['samples'] for row in data) else 'Diagnostic not recorded'
                axis.text(.5, .5, message, ha='center', va='center', transform=axis.transAxes)
                axis.set_xlim(x[0] - .5, x[-1] + .5)
            axis.set_title(f'Member {owner}')
            axis.set_xlabel('Completed episode')
            axis.grid(alpha=.2)
            if field in ('entropy', 'self'):
                axis.set_ylim(-.03, 1.03)
            if field == 'ev':
                axis.set_yscale('symlog', linthresh=1)
        fig.suptitle(f'{label} | per episode + trailing 10 episodes')
        fig.tight_layout()
        fig.savefig(output / f'member_{field}.png', dpi=150)
        plt.close(fig)
    (output / 'aggregation.json').write_text(json.dumps(dict(
        source='completed episodes only', means='weighted by nonmissing diagnostic sample counts',
        ev='median of defined update EVs; smoothing is median of episode medians',
        returns='GAE targets, not episode reward sums',
        missing='blank in CSV, gaps in curves; no samples counted as zero samples only'), indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    print(plot(parser.parse_args().run))
