"""Compare fixed-stream validation delays for the two joint-training methods."""
import argparse
import csv
import json
from pathlib import Path

from scripts.plot_joint_delay import plt, np, read_run


def plot(root):
    root = Path(root).resolve()
    output = root / 'figures'
    output.mkdir(exist_ok=True)
    datasets = [read_run(root, method) for method in ('joint_delay_ppo', 'joint_delay_acac')]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'legend.frameon': False})
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    fig.subplots_adjust(left=.12, right=.97, top=.88, bottom=.25)
    summary, records = {}, []
    for data, name, color, style in zip(datasets, ('PPO', 'ACAC'), ('#356A99', '#C67843'), ('-', '--')):
        x, y = data['val_x'], data['val_y']
        ax.plot(x, y, color=color, ls=style, lw=1.7, marker='o', ms=3, label=name)
        best = int(np.argmin(y))
        ax.scatter(x[best], y[best], s=115, marker='*', color=color,
                   edgecolors='white', linewidths=.6, zorder=5)
        summary[name] = dict(checkpoints=len(x), best_episode=int(x[best]), best_delay_ms=float(y[best]),
                             latest_episode=int(x[-1]), latest_delay_ms=float(y[-1]))
        for row in data['validation']:
            records.append(dict(method=name, **row))
    ax.set(title='Fixed validation streams: PPO vs ACAC', xlabel='Training episode',
           ylabel='Mean DAG completion delay (ms)', xlim=(0, 301))
    ax.set_xticks(np.arange(0, 301, 25))
    ax.grid(axis='y', color='#E7EAED', lw=.6)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right')
    fig.text(.12, .14, 'Mean of the same 3 task streams; 500 slots / 10,000 DAGs per stream. Exploration off.',
             fontsize=7.7, color='#687581')
    fig.text(.12, .09, 'Stars: best validation checkpoints. No smoothing; lower is better. One training seed (45).',
             fontsize=7.7, color='#687581')
    fig.text(.12, .04, 'Best: ' + '; '.join(f'{name} {row["best_delay_ms"]:.2f} ms (episode {row["best_episode"]})'
                                         for name, row in summary.items()), fontsize=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    from matplotlib.text import Text
    clipped = []
    for item in fig.findobj(Text):
        if item.get_text() and item.get_visible():
            box = item.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                clipped.append(item.get_text())
    base = output / 'ppo_acac_validation_delay'
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(base.with_suffix('.' + extension), dpi=300, facecolor='white')
    plt.close(fig)
    with base.with_suffix('.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    streams = {name: sorted({row['epoch'] for row in records if row['method'] == name}) for name in summary}
    common = sorted(set(datasets[0]['val_x']) & set(datasets[1]['val_x']))
    last_common = int(common[-1])
    comparison = {name: float(data['val_y'][data['val_x'] == last_common][0])
                  for name, data in zip(summary, datasets)}
    qa = dict(summary=summary, validation_streams=streams, text_outside_canvas=clipped,
              last_common_episode=last_common, last_common_delay_ms=comparison,
              averaging='arithmetic mean across three fixed equal-sized task streams; no smoothing',
              statistics='single training seed; task streams are not independent training seeds')
    base.with_suffix('.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    plot(parser.parse_args().run_root)
