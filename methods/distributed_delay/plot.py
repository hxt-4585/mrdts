"""Plot saved training/validation curves without running a simulation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(run, output=None):
    run = Path(run)
    output = Path(output) if output is not None else run / 'curves'
    output.mkdir(parents=True, exist_ok=True)
    with (run / 'episodes.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    episodes = np.array([int(row['episode']) for row in rows])
    training = np.array([float(row['mean_delay_s']) for row in rows])
    moving_x, moving_y = [], []
    for index in range(4, len(training)):
        moving_x.append(episodes[index])
        moving_y.append(training[index - 4:index + 1].mean())
    validation = {}
    for path in run.glob('validation_*.json'):
        validation[int(path.stem.split('_')[-1])] = json.loads(path.read_text(encoding='utf-8'))
    vx = np.array(sorted(validation))
    values = np.array([[row['mean_delay_s'] for row in validation[episode]] for episode in vx])
    mean, minimum, maximum = values.mean(1), values.min(1), values.max(1)
    best = np.minimum.accumulate(mean)
    results = json.loads((run / 'test_results.json').read_text(encoding='utf-8'))
    comparison = results['comparison']
    cem = comparison['reference_delay_s']
    selected = results['selected_episode']
    selected_delay = mean[vx == selected][0]

    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .8,
                         'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    blue, orange, dark, gray = '#3976a8', '#c77b36', '#965327', '#747e88'
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.7), gridspec_kw={'width_ratios': [1.45, 1.]})
    fig.subplots_adjust(left=.075, right=.98, bottom=.24, top=.76, wspace=.22)
    left, right = axes
    left.plot(episodes, training, color=blue, alpha=.25, linewidth=1., label='训练原始值（含探索）')
    left.plot(moving_x, moving_y, color=blue, linewidth=1.8, label='训练：5回合移动平均')
    left.plot(vx, mean, 'o-', color=orange, markersize=4, linewidth=1.7, label='验证均值（无探索）')
    left.fill_between(vx, minimum, maximum, color=orange, alpha=.13)
    left.step(vx, best, where='post', color=dark, linestyle='--', linewidth=1.25, label='历史最佳验证')
    left.axhline(cem, color=gray, linestyle=':', linewidth=1.7, label='CEM 测试参考')
    left.set(title='a  完整训练过程', xlabel='训练回合', ylabel='平均 DAG 时延（秒）',
             xlim=(0, episodes[-1] + 1), ylim=(cem - .012, max(training.max(), maximum.max()) + .02))
    left.legend(loc='upper right', fontsize=8.3)

    late = vx >= 20
    right.fill_between(vx[late], minimum[late], maximum[late], color=orange, alpha=.13)
    right.plot(vx[late], mean[late], 'o-', color=orange, markersize=4.5, linewidth=1.7, label='当次验证')
    right.step(vx[late], best[late], where='post', color=dark, linestyle='--', linewidth=1.3, label='历史最佳验证')
    right.axhline(cem, color=gray, linestyle=':', linewidth=1.7, label=f'CEM 测试：{cem:.5f} 秒')
    right.scatter([selected], [selected_delay], marker='*', s=140, color=dark, edgecolors='white', linewidths=.8, zorder=5)
    right.annotate(f'选中第 {selected} 回合\n验证 {selected_delay:.5f} 秒',
                   xy=(selected, selected_delay), xytext=(selected - 3, selected_delay - .006),
                   ha='right', va='top', fontsize=9, color=dark,
                   arrowprops={'arrowstyle': '-', 'color': dark, 'lw': .8})
    right.set(title='b  验证后期放大', xlabel='训练回合', ylabel='平均 DAG 时延（秒）',
              xlim=(18, episodes[-1] + 2), ylim=(cem - .004, max(maximum[late]) + .01))
    right.legend(loc='upper left', fontsize=8.3)
    for axis in axes:
        axis.grid(axis='y', color='#d7dce1', alpha=.65, linewidth=.65)
        axis.set_axisbelow(True)
        axis.tick_params(direction='out')
        axis.ticklabel_format(axis='y', style='plain', useOffset=False)
    fig.suptitle('分布式时延优化：训练与验证曲线', x=.075, y=.95, ha='left', fontsize=16, fontweight='bold')
    fig.text(.075, .855, '冻结低层；每时隙3轮优化；60回合 × 500时隙。回合0为未训练代理的局部覆盖初始化。',
             fontsize=10, color='#4d5660')
    fig.text(.075, .135, f'独立配对测试：分布式 {comparison["learned_delay_s"]:.5f} 秒；'
             f'CEM {cem:.5f} 秒；时延差 +{comparison["relative_delay_difference"] * 100:.2f}%（10回合）。',
             fontsize=10, color='#303740')
    fig.text(.075, .075, '阴影为3个验证回合的最小–最大范围，非置信区间。训练、验证、测试的任务流不同；CEM横线仅作参考。',
             fontsize=9, color='#5e6770')
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'delay_curves.{extension}', dpi=220, facecolor='white')
    plt.close(fig)

    with (output / 'source_data.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['series', 'episode', 'mean_delay_s', 'minimum_s', 'maximum_s'])
        writer.writerows(['training', int(x), float(y), '', ''] for x, y in zip(episodes, training))
        writer.writerows(['training_trailing_mean_5', int(x), float(y), '', ''] for x, y in zip(moving_x, moving_y))
        writer.writerows(['validation', int(x), float(y), float(lo), float(hi)]
                         for x, y, lo, hi in zip(vx, mean, minimum, maximum))
        writer.writerows(['best_validation', int(x), float(y), '', ''] for x, y in zip(vx, best))
        writer.writerow(['cem_held_out_test', '', cem, '', ''])
        writer.writerow(['distributed_held_out_test', selected, comparison['learned_delay_s'], '', ''])
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(plot(args.run, args.output))
