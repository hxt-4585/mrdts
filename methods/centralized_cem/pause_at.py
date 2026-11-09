"""One scheduled boundary check; suspend this training tree and draw diagnostics."""

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess


def suspend(pid, resume=False):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x0800, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        ntdll = ctypes.WinDLL('ntdll')
        operation = ntdll.NtResumeProcess if resume else ntdll.NtSuspendProcess
        operation.argtypes = [wintypes.HANDLE]
        operation.restype = ctypes.c_long
        result = operation(handle)
        if result != 0:
            raise OSError(f'Process operation failed for {pid}: NTSTATUS {result}')
    finally:
        kernel.CloseHandle(handle)


def check(output, generation, task_name, monitor_task):
    metadata_path = output / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata['status'] == 'running':
        if metadata['generation'] < generation or metadata['phase'] != 'search':
            return
        if (output / 'checkpoint.tmp').exists() or (output / 'search_state.tmp').exists():
            return
        pid = metadata['pid']
        command = ('Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine,CreationDate '
                   '| ConvertTo-Json -Compress')
        processes = json.loads(subprocess.check_output(
            ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command], text=True))
        parent = next(p for p in processes if p['ProcessId'] == pid)
        if 'methods.centralized_cem.run' not in parent['CommandLine']:
            raise RuntimeError('Training process identity changed')
        descendants = {pid}
        for _ in processes:
            expanded = descendants | {p['ProcessId'] for p in processes if p['ParentProcessId'] in descendants}
            if expanded == descendants:
                break
            descendants = expanded
        suspended = []
        try:
            for process_id in [pid, *sorted(descendants - {pid})]:
                suspend(process_id)
                suspended.append(process_id)
            # Read the stable state again after suspension.
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            pause = dict(paused_at=datetime.now(timezone.utc).isoformat(), requested_generation=generation,
                         generation=metadata['generation'], suspended_pids=suspended,
                         processes=[p for p in processes if p['ProcessId'] in descendants],
                         resume='NtResumeProcess on all suspended_pids; or launch --resume-from this directory',
                         previous_metadata=metadata.copy())
            (output / 'pause.json').write_text(json.dumps(pause, indent=2), encoding='utf-8')
            snapshot = output / f'paused_gen{metadata["generation"]}'
            snapshot.mkdir(exist_ok=True)
            for name in ('checkpoint.pt', 'search_state.json', 'config.json', 'frozen_members.pt'):
                shutil.copy2(output / name, snapshot / name)
            metadata.update(status='paused', paused_at=pause['paused_at'])
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        except BaseException:
            for process_id in reversed(suspended):
                suspend(process_id, resume=True)
            raise
    if metadata['status'] == 'paused':
        from methods.centralized_cem.plot import plot
        plot(output, generation)
        for name in (monitor_task, task_name):
            subprocess.run(['schtasks', '/Change', '/TN', name, '/DISABLE'], check=True, capture_output=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generation', type=int, default=100)
    parser.add_argument('--task-name', required=True)
    parser.add_argument('--monitor-task', required=True)
    args = parser.parse_args()
    check(args.output, args.generation, args.task_name, args.monitor_task)
