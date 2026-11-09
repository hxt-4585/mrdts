"""Frozen ACAC validation with actual per-UAV horizontal flight velocity."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from methods.joint_delay_acac.learning import make_model
from methods.joint_delay_acac.planner import Controller, observe
from methods.joint_delay_acac.rollout import collect_slot
from methods.joint_delay_acac.support import aggregate, make_environment, write_json


def draw(output, speed, config, epoch, selected):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'legend.frameon': False})
    slots, members = speed.shape
    x = np.arange(1, slots + 1)
    limit = np.sqrt(2) * config.max_horizontal_speed
    mean = speed.mean(axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    fig.subplots_adjust(left=.11, right=.98, bottom=.22, top=.88)
    ax.fill_between(x, speed.min(axis=1), speed.max(axis=1), color='#356A99', alpha=.15,
                    label='Minimum to maximum across UAVs')
    ax.plot(x, mean, color='#356A99', lw=1.6, label='Mean across 12 UAVs')
    ax.axhline(limit, color='#C67843', lw=.9, ls='--', label=f'Component-wise limit: {limit:.2f} m/s')
    ax.set(xlim=(1, max(2, slots)), ylim=(-.3, limit + 1), xlabel='Scheduling slot',
           ylabel='Horizontal flight speed (m/s)', title='ACAC: actual UAV flight speed during validation')
    ax.grid(axis='y', color='#E7EAED', lw=.6)
    ax.legend(loc='upper right', fontsize=8, frameon=True, facecolor='white', framealpha=1., edgecolor='none')
    fig.text(.11, .09, f'Frozen checkpoint: episode {selected}; validation stream {epoch}; exploration off.', fontsize=8)
    fig.text(.11, .035, 'Speed = displacement / 0.5 s flight phase; scheduling-phase hover is not included.',
             fontsize=7.8, color='#687581')
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'fleet_speed.{extension}', dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(10.2, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.98, bottom=.12, top=.91, hspace=.32, wspace=.15)
    for owner, ax in enumerate(axes.flat):
        ax.plot(x, speed[:, owner], color='#356A99', lw=.9)
        ax.set_title(f'UAV {owner + 1}', fontsize=9, loc='left')
        ax.set(xlim=(1, max(2, slots)), ylim=(-.3, limit + 1))
        ax.grid(axis='y', color='#E7EAED', lw=.5)
        ax.tick_params(labelsize=8)
    fig.suptitle('ACAC validation: per-UAV horizontal flight speed', fontsize=13, x=.075, ha='left')
    fig.supxlabel('Scheduling slot', fontsize=10, y=.065)
    fig.supylabel('Horizontal flight speed (m/s)', fontsize=10, x=.015)
    fig.text(.075, .02, f'Checkpoint episode {selected}; {slots} slots; one fixed validation task stream; no fitting or exploration.',
             fontsize=8, color='#687581')
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'uav_speeds.{extension}', dpi=300)
    plt.close(fig)


def run(checkpoint, output, *, epoch=1500000, slots=500):
    torch.set_num_threads(1)
    raw = Path(checkpoint).read_bytes()
    saved = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    config = saved['config']
    env = make_environment(saved['environment'])
    count = env.members.member_uav_count
    model = make_model(count, 5 * count + 8, config['train'], 'cpu')
    model.load_state_dict(saved['low'])
    model.eval()
    controller = Controller(observe(env), config['high'])
    controller.load_policy([(state['model'], state['samples']) for state in saved['high']])
    env.reset(epoch_index=epoch)
    controller.reset(epoch)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metadata = dict(status='running', pid=os.getpid(), checkpoint=str(Path(checkpoint).resolve()),
                    checkpoint_sha256=hashlib.sha256(raw).hexdigest(), selected_episode=saved['selected_episode'],
                    epoch=epoch, slots_planned=slots, slots_completed=0, device='cpu',
                    started_at=datetime.now(timezone.utc).isoformat())

    def status(**changes):
        metadata.update(changes, elapsed_s=time.perf_counter() - started,
                        updated_at=datetime.now(timezone.utc).isoformat())
        write_json(output / 'metadata.json', metadata)

    rows, speeds = [], []
    status()
    try:
        fields = ['slot', 'uav', 'x_before_m', 'y_before_m', 'x_after_m', 'y_after_m',
                  'vx_mps', 'vy_mps', 'speed_mps', 'command_vx_mps', 'command_vy_mps']
        with (output / 'velocities.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for slot in range(slots):
                before = env.members.positions[:count, :2].astype(np.float64)
                actions = controller.actions(observe(env), 0.)
                _, row, _ = collect_slot(env, model, actions, config['train'], deterministic=True)
                after = env.members.positions[:count, :2].astype(np.float64)
                velocity = (after - before) / env.env_config.flight_duration_s
                speed = np.linalg.norm(velocity, axis=1)
                speeds.append(speed)
                rows.append(row)
                commanded = actions * env.members.config.max_horizontal_speed
                for owner in range(count):
                    writer.writerow(dict(slot=slot + 1, uav=owner + 1,
                        x_before_m=before[owner, 0], y_before_m=before[owner, 1],
                        x_after_m=after[owner, 0], y_after_m=after[owner, 1],
                        vx_mps=velocity[owner, 0], vy_mps=velocity[owner, 1], speed_mps=speed[owner],
                        command_vx_mps=commanded[owner, 0], command_vy_mps=commanded[owner, 1]))
                if slot == 4 or (slot + 1) % 25 == 0 or slot + 1 == slots:
                    stream.flush()
                    elapsed = time.perf_counter() - started
                    status(slots_completed=slot + 1, estimated_seconds_remaining=elapsed / (slot + 1) * (slots - slot - 1))
            velocity_data = np.asarray(speeds)
        status(phase='plotting')
        draw(output, velocity_data, env.members.config, epoch, saved['selected_episode'])
        summary = dict(**aggregate(rows), selected_episode=saved['selected_episode'], epoch=epoch,
                       mean_flight_speed_mps=float(velocity_data.mean()), max_flight_speed_mps=float(velocity_data.max()),
                       per_uav_mean_mps=velocity_data.mean(axis=0).tolist(),
                       fraction_speed_below_0_1_mps=float((velocity_data < .1).mean()),
                       definition='Actual displacement divided by 0.5 s flight phase, not full-slot average',
                       frozen=True, exploration=0., members=count)
        write_json(output / 'summary.json', summary)
        status(status='completed', phase='done')
    except BaseException as error:
        status(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epoch', type=int, default=1500000)
    parser.add_argument('--slots', type=int, default=500)
    args = parser.parse_args()
    print(run(args.checkpoint, args.output, epoch=args.epoch, slots=args.slots))
