"""One bounded inspection, run by Windows Task Scheduler every fifteen minutes."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def replace_file(source, target):
    """Keep monitor output atomic while tolerating a short-lived open reader."""
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.02 * 2 ** attempt, .5))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    replace_file(temporary, path)


def process_alive(pid):
    result = subprocess.run(['tasklist.exe', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
                            capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    return any(len(row) > 1 and row[1] == str(pid) for row in csv.reader(io.StringIO(result.stdout)))


def paired_comparison(runs):
    reports = {name: json.loads((Path(run['output']) / 'test_results.json').read_text(encoding='utf-8'))
               for name, run in runs.items()}
    ppo = {r['epoch']: r for r in reports['joint_delay_ppo']['episodes']}
    acac = {r['epoch']: r for r in reports['joint_delay_acac']['episodes']}
    epochs = sorted(ppo)
    reference = np.asarray([ppo[e]['mean_delay_s'] for e in epochs])
    candidate = np.asarray([acac[e]['mean_delay_s'] for e in epochs])
    indices = np.random.default_rng(45).integers(len(epochs), size=(10000, len(epochs)))
    return dict(epochs=epochs, ppo_delay_s=float(reference.mean()), acac_delay_s=float(candidate.mean()),
                relative_delay_difference=float(candidate.mean() / reference.mean() - 1),
                paired_difference_95_ci_s=np.quantile((candidate - reference)[indices].mean(1), [.025, .975]).tolist(),
                ppo_failed_dags=sum(ppo[e]['failed_dags'] for e in epochs),
                acac_failed_dags=sum(acac[e]['failed_dags'] for e in epochs),
                interpretation='one training seed and fixed scene; paired task-stream uncertainty only')


def inspect(manifest_path, *, disable_task=True):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), status='running', runs={})
    terminal = []
    successful = []
    for name, run in manifest['runs'].items():
        directory = Path(run['output'])
        alive = process_alive(run['pid'])
        metadata_path = directory / 'metadata.json'
        info = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else dict(status='starting')
        snapshot = dict(info, process_alive=alive, output=str(directory), stdout=run.get('stdout'), stderr=run.get('stderr'))
        done = info['status'] in ('completed', 'failed')
        if not done and not alive:
            snapshot.update(status='failed', attention='Process exited; inspect stderr and latest.pt.')
            done = True
        config_path = directory / 'config.json'
        if config_path.exists():
            train = json.loads(config_path.read_text(encoding='utf-8'))['config']['train']
            snapshot['total_training_slots'] = train['episodes'] * train['slots']
            completed_slots = info.get('slots_completed', 0)
            if completed_slots:
                snapshot['estimated_training_hours_remaining'] = max(0.,
                    (snapshot['total_training_slots'] - completed_slots) * info['elapsed_s'] / completed_slots / 3600)
        result['runs'][name] = snapshot
        terminal.append(done)
        successful.append(info['status'] == 'completed')
    if all(terminal):
        result['status'] = 'completed' if all(successful) else 'failed'
        if all(successful):
            result['comparison'] = paired_comparison(manifest['runs'])
            write_json(manifest_path.parent / 'comparison.json', result['comparison'])
        if disable_task:
            disabled = subprocess.run(['schtasks.exe', '/Change', '/TN', manifest['task_name'], '/DISABLE'],
                                     capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            result['task_disabled'] = disabled.returncode == 0
        write_json(manifest_path.parent / 'completion.json', result)
    write_json(manifest_path.parent / 'monitor.json', result)
    with (manifest_path.parent / 'monitor_history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.manifest), ensure_ascii=False))
