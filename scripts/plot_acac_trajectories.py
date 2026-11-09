"""Plot actual XY trajectories from a frozen ACAC velocity-validation capture."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch

from methods.joint_delay_acac.support import make_environment


def plot(output):
    output = Path(output).resolve()
    meta = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
    source = torch.load(meta['checkpoint'], map_location='cpu', weights_only=True)
    env = make_environment(source['environment'])
    with (output / 'velocities.csv').open(encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    count = env.members.member_uav_count
    paths = []
    for owner in range(1, count + 1):
        rows = [row for row in records if int(row['uav']) == owner]
        points = [[float(rows[0]['x_before_m']), float(rows[0]['y_before_m'])]]
        points.extend([float(row['x_after_m']), float(row['y_after_m'])] for row in rows)
        paths.append(np.asarray(points))
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'legend.frameon': False})
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, count))
    side = env.region.config.side_length
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig.subplots_adjust(left=.11, right=.96, top=.90, bottom=.25)
    regions = env.region.region_map
    cell = side / env.region.config.grid_size
    vertical = np.argwhere(regions[:, :-1] != regions[:, 1:])
    horizontal = np.argwhere(regions[:-1] != regions[1:])
    borders = [[((col + 1) * cell, row * cell), ((col + 1) * cell, (row + 1) * cell)]
               for row, col in vertical]
    borders.extend([[(col * cell, (row + 1) * cell), ((col + 1) * cell, (row + 1) * cell)]
                    for row, col in horizontal])
    ax.add_collection(LineCollection(borders, colors='#CED4DA', linewidths=.7))
    ax.scatter(env.users.positions[:, 0], env.users.positions[:, 1], s=10, color='#9EA6AD', alpha=.5, zorder=1)
    for index, (path, color) in enumerate(zip(paths, colors)):
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.35, label=f'UAV {index + 1}', zorder=2)
        ax.scatter(*path[0], color=color, marker='o', s=25, facecolors='none', linewidths=1.1, zorder=3)
        ax.scatter(*path[-1], color=color, marker='x', s=28, linewidths=1.2, zorder=4)
    ax.set(xlim=(0, side), ylim=(0, side), xlabel='X position (m)', ylabel='Y position (m)',
           title='ACAC validation: UAV flight trajectories', aspect='equal')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=4, fontsize=8, columnspacing=1.4)
    fig.text(.11, .07, 'Open circle: start; cross: end. Gray dots: users; gray lines: region boundaries.', fontsize=8)
    fig.text(.11, .03, f'Frozen episode {meta["selected_episode"]}; validation stream {meta["epoch"]}; '
             f'{len(paths[0]) - 1} slots; exploration off.', fontsize=8, color='#687581')
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'uav_trajectories.{extension}', dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(10.2, 7.5))
    fig.subplots_adjust(left=.075, right=.98, bottom=.13, top=.91, hspace=.48, wspace=.32)
    for owner, (ax, path, color) in enumerate(zip(axes.flat, paths, colors)):
        center = (path.max(0) + path.min(0)) * .5
        radius = np.ptp(path, axis=0).max() * .5 + 10
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.1)
        ax.scatter(*path[0], marker='o', facecolors='none', edgecolors=[color], s=27)
        ax.scatter(*path[-1], marker='x', color=color, s=25)
        ax.set(title=f'UAV {owner + 1}', aspect='equal', xlim=(center[0] - radius, center[0] + radius),
               ylim=(center[1] - radius, center[1] + radius))
        ax.tick_params(labelsize=7)
        ax.locator_params(axis='both', nbins=3)
        ax.grid(color='#E7EAED', lw=.5)
    fig.suptitle('ACAC validation: individual UAV trajectories', fontsize=13, x=.075, ha='left')
    fig.supxlabel('X position (m)', fontsize=10, y=.065)
    fig.supylabel('Y position (m)', fontsize=10, x=.015)
    fig.text(.075, .02, 'Open circle: start; cross: end. Each panel uses local spatial limits; no smoothed or invented positions.',
             fontsize=8, color='#687581')
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'individual_trajectories.{extension}', dpi=300)
    plt.close(fig)
    initial = np.stack([path[0] for path in paths])
    qa = dict(uavs=count, slots=len(paths[0]) - 1,
              initial_positions_match_snapshot=bool(np.allclose(initial, env.members.positions[:count, :2])),
              per_uav_path_length_m=[float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) for path in paths],
              selected_episode=meta['selected_episode'], epoch=meta['epoch'])
    (output / 'trajectory_summary.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    plot(parser.parse_args().output)
