"""Plot recorded CEM diagnostics, without rerunning any simulation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(output, through=100):
    output = Path(output)
    directories = []
    directory = output
    while directory is not None:
        directories.append(directory)
        config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        directory = Path(config['resume_from']) if config.get('resume_from') else None
    rows, validation = {}, {}
    for directory in reversed(directories):
        with (directory / 'generations.csv').open(encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                generation = int(row['generation'])
                if generation <= through and row.get('elapsed_s'):
                    rows[generation] = row
        for path in directory.glob('validation_*.json'):
            generation = int(path.stem.split('_')[1])
            if generation <= through:
                validation[generation] = json.loads(path.read_text(encoding='utf-8'))
    x = np.array(sorted(rows))
    best = np.array([float(rows[g]['best_train_delay_s']) for g in x])
    population = np.array([float(rows[g]['population_mean_delay_s']) for g in x])
    vx = np.array(sorted(validation))
    delays = np.array([[r['mean_delay_s'] for r in validation[g]] for g in vx])
    failures = [100 * sum(r['failed_dags'] for r in validation[g]) /
                sum(r['dag_count'] for r in validation[g]) for g in vx]
    violations = [sum(r['flight_violations'] for r in validation[g]) for g in vx]
    charts = output / f'curves_gen{through}'
    charts.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    blue, orange, gray = '#2563eb', '#c65d15', '#94a3b8'
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), layout='constrained')
    ax = axes[0, 0]
    ax.plot(x, -population, color=gray, label='种群平均（训练）')
    ax.plot(x, -best, color=blue, label='历史最优（训练）')
    ax.plot(vx, -delays.mean(axis=1), 'o-', ms=3, color=orange, label='候选最优部署的验证表现')
    ax.set(title='A  Reward：越高越好', ylabel='−平均 DAG 时延（秒）')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    means = delays.mean(axis=1)
    ax.plot(vx, means, 'o-', ms=3, color=orange, label='验证均值')
    ax.fill_between(vx, delays.min(axis=1), delays.max(axis=1), color=orange,
                    alpha=.15, label='3轮验证的最小–最大值')
    ax.set(title='B  验证时延：越低越好', ylabel='平均 DAG 时延（秒）')
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    late = x >= 60
    vl = vx >= 60
    ax.plot(x[late], best[late], color=blue, label='训练最佳')
    ax.plot(vx[vl], means[vl], 'o-', ms=3, color=orange, label='当次验证')
    ax.set(title='C  第60代以后：时延放大图', ylabel='平均 DAG 时延（秒）')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(x, (population - best) * 1000, color=blue)
    ax.set(title='D  当代种群均值与历史最优的差距', ylabel='时延差（毫秒）')
    ax = axes[2, 0]
    ax.plot(x, [float(rows[g]['search_std_mean']) * 1000 for g in x], color=blue)
    ax.axhline(5, color=gray, ls='--', label='配置下限：5米')
    ax.set(title='E  搜索分布的平均坐标标准差', ylabel='标准差（米，地图边长1000米）')
    ax.legend(fontsize=8)
    ax = axes[2, 1]
    ax.plot(vx, failures, 'o-', ms=3, color=orange, label='DAG失败率（%）')
    ax.plot(vx, violations, 'x--', ms=4, color=blue, label='越界次数')
    ax.set(title='F  验证可靠性', ylabel='失败率（%） / 越界次数')
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('搜索代数')
        ax.grid(alpha=.18)
        if ax is not axes[1, 0]:
            ax.axvline(60, color=gray, lw=.8, ls=':')
    fig.suptitle(f'集中式 CEM + 冻结低层：截至第{x[-1]}代的搜索诊断\n'
                 '训练128时隙；验证500时隙 × 3轮。任务流不同，二者绝对值不可直接比较。', fontsize=13)
    fig.savefig(charts / 'diagnostics.png', dpi=160)
    fig.savefig(charts / 'diagnostics.pdf')
    plt.close(fig)
    # A separate zoom avoids compressing the small late-stage improvements.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    axes[0].plot(x[late], best[late], color=blue, marker='.', ms=3)
    axes[1].plot(vx[vl], means[vl], 'o-', color=orange, ms=4, label='当次验证')
    axes[1].plot(vx[vl], np.minimum.accumulate(means)[vl], '--', color=gray, label='历史最佳验证')
    axes[0].set_title('训练最佳时延（第60代以后）')
    axes[1].set_title('验证时延（第60代以后）')
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set(xlabel='搜索代数', ylabel='平均 DAG 时延（秒）')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.grid(alpha=.2)
    fig.savefig(charts / 'late_stage.png', dpi=160)
    plt.close(fig)
    with (charts / 'source_data.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[x[0]]))
        writer.writeheader()
        writer.writerows(rows[g] for g in x)
    summary = dict(generation=int(x[-1]), train_delay_s=float(best[-1]),
        best_validation_delay_s=float(means.min()), latest_validation_delay_s=float(means[-1]),
        latest_validation_generation=int(vx[-1]), latest_validation_failure_rate_percent=failures[-1],
        latest_validation_violations=violations[-1],
        train_improvement_since_60_percent=100*(float(rows[60]['best_train_delay_s'])-best[-1])/float(rows[60]['best_train_delay_s']),
        validation_improvement_since_60_percent=100*(means[vx == 60][0]-means.min())/means[vx == 60][0])
    (charts / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return charts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--through', type=int, default=100)
    args = parser.parse_args()
    print(plot(args.output, args.through))
