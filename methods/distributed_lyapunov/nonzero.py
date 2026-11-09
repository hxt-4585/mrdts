"""Frozen-policy budget sweep and continuous nonzero-queue verification.

Only the snapshot's UAV slot budget changes. No artificial queue injection,
clipping, resetting within a trajectory, or policy updates are performed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from env.env import MRDTSEnv
from .learning import MemberPolicy
from .planner import frames
from .simulation import execute, queue_statistics
from .run import write_json


def nonzero_statistics(queues, energies, budget):
    q, e = np.asarray(queues), np.asarray(energies)
    tail = q[len(q)//2:]
    tail_energy = e[len(q)//2:]
    blocks = np.array_split(tail, 4)
    means = np.array([block.mean(0) for block in blocks])
    fractions = np.array([(block > 1e-6).mean(0) for block in blocks])
    stats = queue_statistics(q, e, budget)
    tail_stats = queue_statistics(tail, tail_energy, budget)
    persistent = (fractions.min(0) >= .05) & (tail.mean(0) >= 1e-5*budget)
    # Do not infer stationarity from whole-horizon averages or Q(T)/T alone.
    plateau = np.abs(means[-1]-means[-2]) <= np.maximum(.1, .3*means[-2])
    bounded = (np.asarray(tail_stats['tail_slope_j_per_slot']) <= 1e-5*budget)
    energy_ok = tail_energy.mean(0) <= budget + 1e-5*budget
    terminal_ok = q[-1]/len(q) <= 1e-4*budget
    stable = bool(np.all(plateau & bounded & energy_ok & terminal_ok))
    stats.update(nonzero_stable=bool(stable and persistent.any()),
                 stricter_tail_stable=stable, persistent_uav_ids=np.flatnonzero(persistent).tolist(),
                 tail_nonzero_fraction=(tail > 1e-6).mean(0).tolist(),
                 tail_mean_queue_j=tail.mean(0).tolist(), tail_block_means_j=means.tolist(),
                 tail_block_nonzero_fractions=fractions.tolist(),
                 tail_mean_energy_j=tail_energy.mean(0).tolist(),
                 last_quarter_slope_j_per_slot=tail_stats['tail_slope_j_per_slot'],
                 plateau_per_uav=plateau.tolist(), slots=len(q))
    return stats


def evaluate(job):
    torch.set_num_threads(1)
    folder = Path(job['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    summary = folder/'summary.json'
    if summary.exists():
        return json.loads(summary.read_text(encoding='utf-8'))
    source = torch.load(job['checkpoint'], weights_only=False, map_location='cpu')
    settings = dict(source['environment'])
    settings['uav'] = replace(settings['uav'], member_slot_energy=job['budget'])
    env = MRDTSEnv(settings)
    env.generate_scene()
    env.reset(epoch_index=job['epoch'])
    policies = nn.ModuleList([MemberPolicy(env.members.member_uav_count)
                             for _ in range(env.members.member_uav_count)])
    policies.load_state_dict(source['policies'])
    policies.eval()
    controller = source['controller']
    queues, energies, delay, failure = [], [], 0., 0.
    horizons = {}
    started = perf_counter()
    with (folder/'slots.jsonl').open('w', encoding='utf-8') as stream:
        for slot in range(job['slots']):
            actions, _ = controller.actions(frames(env))
            _, row, _ = execute(env, policies, actions, job['v'], deterministic=True)
            row.update(slot=slot, epoch=job['epoch'])
            stream.write(json.dumps(row)+'\n')
            queues.append(row['queue_j'])
            energies.append(row['energy_j'])
            delay += row['mean_delay_s']
            failure += row['failure_rate']
            if (slot+1) % 5000 == 0 or slot+1 in [1000,2000,3000] or slot+1 == job['slots']:
                horizons[str(slot+1)] = nonzero_statistics(queues, energies, job['budget'])
                stream.flush()
                write_json(folder/'progress.json', dict(job, slots_completed=slot+1,
                           elapsed_s=perf_counter()-started, horizons=horizons, pid=os.getpid()))
    result = dict(job, **nonzero_statistics(queues, energies, job['budget']), horizons=horizons,
                  mean_delay_s=delay/job['slots'], failure_rate=failure/job['slots'],
                  wall_seconds=perf_counter()-started,
                  interpretation='Frozen trained-at-220J policies transferred to a new budget; finite-horizon evidence only.')
    write_json(summary, result)
    return result


def run_jobs(jobs, output, phase):
    results = []
    write_json(output/'status.json', dict(state=phase, pid=os.getpid(), completed=0, total=len(jobs)))
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(evaluate, job):job for job in jobs}
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            write_json(output/f'{phase}.json', results)
            write_json(output/'status.json', dict(state=phase, pid=os.getpid(),
                       completed=len(results), total=len(jobs), latest=dict(
                       mode=result['mode'], budget=result['budget'], nonzero_stable=result['nonzero_stable'])))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--screen-slots', type=int, default=3000)
    parser.add_argument('--confirm-slots', type=int, default=20000)
    parser.add_argument('--budget', type=float)
    parser.add_argument('--modes', nargs='+', choices=['joint','staged'], default=['joint','staged'])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'status.json', dict(state='screen', pid=os.getpid(),
               source=str(args.source.resolve()), screen_slots=args.screen_slots,
               confirm_slots=args.confirm_slots))
    try:
        candidates = []
        if args.budget is None:
            jobs = []
            for mode in args.modes:
                for budget in [145.,130.,129.,128.8,128.6,128.5,128.45,128.4,128.3]:
                    jobs.append(dict(mode=mode, budget=budget, v=100000., epoch=3500000,
                                slots=args.screen_slots, checkpoint=str((args.source/mode/'selected.pt').resolve()),
                                folder=str(output/'screen'/mode/str(budget))))
            screens = run_jobs(jobs, output, 'screen')
            for mode in args.modes:
                viable = [r for r in screens if r['mode']==mode and r['nonzero_stable']]
                if viable:
                    # Prefer substantial tail occupancy over a tiny initial excursion.
                    candidates.append(max(viable, key=lambda r: max(r['tail_nonzero_fraction'])))
        else:
            candidates = [dict(mode=mode, budget=args.budget, v=100000.,
                              checkpoint=str((args.source/mode/'selected.pt').resolve()),
                              selection='Direct long-horizon diagnostic; short screening did not establish stability.')
                          for mode in args.modes]
        write_json(output/'candidates.json', candidates)
        if not candidates:
            write_json(output/'status.json', dict(state='needs_analysis', pid=os.getpid(),
                       reason='No screened policy has persistent nonzero, bounded queues. Inspect raw traces before adapting training.'))
            return
        confirmations = []
        for candidate in candidates:
            for epoch in [4500000,4500001,4500002]:
                confirmations.append(dict(mode=candidate['mode'], budget=candidate['budget'],
                            v=candidate['v'], epoch=epoch, slots=args.confirm_slots,
                            checkpoint=candidate['checkpoint'], folder=str(output/'confirm'/candidate['mode']/str(epoch))))
        results = run_jobs(confirmations, output, 'confirm')
        passed = [mode for mode in ['joint','staged'] if
                  len([r for r in results if r['mode']==mode])==3 and
                  all(r['nonzero_stable'] for r in results if r['mode']==mode)]
        write_json(output/'status.json', dict(state='complete' if passed else 'needs_analysis',
                   pid=os.getpid(), verified_modes=passed,
                   reason='All three independent continuous horizons must pass persistent nonzero and bounded-tail criteria.'))
    except Exception as error:
        write_json(output/'status.json', dict(state='failed', pid=os.getpid(), error=repr(error)))
        raise


if __name__ == '__main__':
    main()
