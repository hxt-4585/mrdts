"""A one-shot Windows Task Scheduler monitor; never changes the learner.

python -B -m methods.centralized_ppo.monitor RUN --task-name TASK
Writes monitor.json/history.jsonl, plus completion.txt when training finishes.
"""

import argparse
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def process_alive(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = wintypes.DWORD()
    kernel.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel.CloseHandle(handle)
    return code.value == 259


def inspect_run(run, task_name=None):
    metadata = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    snapshot = dict(checked_at=datetime.now(timezone.utc).isoformat(),
                    training_status=metadata['status'], process_alive=process_alive(metadata['pid']),
                    phase=metadata.get('phase'), iteration=metadata['iteration'],
                    planned_updates=config['updates'],
                    best_validation_delay_s=metadata.get('best_validation_delay_s'),
                    metadata_age_s=datetime.now(timezone.utc).timestamp() - (run / 'metadata.json').stat().st_mtime)
    csv_path = run / 'updates.csv'
    if csv_path.exists():
        with csv_path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            snapshot['last_completed_iteration'] = int(rows[-1]['iteration'])
            snapshot['latest_update'] = rows[-1]
            if len(rows) > 1:
                recent = rows[-6:]
                seconds_per_update = ((float(recent[-1]['elapsed_s']) - float(recent[0]['elapsed_s'])) /
                                      (int(recent[-1]['iteration']) - int(recent[0]['iteration'])))
                snapshot['estimated_training_hours_remaining'] = ((config['updates'] - int(rows[-1]['iteration'])) *
                                                                   seconds_per_update / 3600)
    terminal = metadata['status'] in ('completed', 'failed', 'interrupted')
    if not terminal and not snapshot['process_alive']:
        snapshot['attention'] = 'Training process exited before a terminal metadata status; inspect stderr log.'
    elif not terminal and snapshot['metadata_age_s'] > 7200:
        snapshot['attention'] = 'No metadata update in two hours; inspect process and workload.'
    result = run / 'test_results.json'
    if result.exists():
        snapshot['comparisons'] = json.loads(result.read_text(encoding='utf-8'))['comparisons']
    if terminal:
        (run / 'completion.txt').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
        if task_name is not None:
            disabled = subprocess.run(['schtasks.exe', '/Change', '/TN', task_name, '/DISABLE'],
                                      capture_output=True, text=True)
            snapshot['monitor_task_disabled'] = disabled.returncode == 0
    temporary = run / 'monitor.tmp'
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(run / 'monitor.json')
    with (run / 'monitor_history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(snapshot, ensure_ascii=False) + '\n')
    return snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--task-name')
    args = parser.parse_args()
    inspect_run(args.run, args.task_name)
