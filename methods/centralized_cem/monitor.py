"""One bounded check, invoked by Windows Task Scheduler every 30 minutes."""

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
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = wintypes.DWORD()
    try:
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def check(output, task_name):
    metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
    alive = process_alive(metadata['pid'])
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(), process_alive=alive, **metadata)
    csv_path = output / 'generations.csv'
    if csv_path.exists():
        with csv_path.open(newline='', encoding='utf-8') as stream:
            rows = [row for row in csv.DictReader(stream) if row.get('elapsed_s')]
        if rows:
            report['last_generation'] = rows[-1]
            report['recent_generations'] = rows[-10:]
    if metadata['status'] == 'running' and not alive:
        report['monitor_status'] = 'process_exited_without_completion'
    else:
        report['monitor_status'] = metadata['status']
    result = output / 'test_results.json'
    if result.exists():
        report['final_test'] = json.loads(result.read_text(encoding='utf-8'))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    (output / 'monitor_latest.json').write_text(text, encoding='utf-8')
    with (output / 'monitor_history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False) + '\n')
    # Monitoring stops when this run terminates; never stop another training job.
    if metadata['status'] != 'running' or not alive:
        subprocess.run(['schtasks', '/Change', '/TN', task_name, '/DISABLE'], check=True,
                       capture_output=True)
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task-name', required=True)
    args = parser.parse_args()
    check(args.output, args.task_name)
