"""One-shot Windows scheduled monitor for this baseline's background evaluation."""

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def process_alive(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


def snapshot(output, alive):
    output = Path(output)
    progress_file = output / "progress.json"
    failure_file = output / "failure.json"
    progress = json.loads(progress_file.read_text(encoding="utf-8")) if progress_file.exists() else dict(status="starting")
    if failure_file.exists():
        progress = dict(status="failed", detail=json.loads(failure_file.read_text(encoding="utf-8")))
    elif progress["status"] != "completed" and not alive:
        progress = {**progress, "status": "failed", "detail": "Process exited before evaluation completed."}
    jobs = [json.loads(file.read_text(encoding="utf-8")) for file in sorted(output.glob("progress_*_*.json"))]
    return {**progress, "process_alive": alive, "jobs": jobs,
            "checked_utc": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    launch = json.loads((args.output / "launch.json").read_text(encoding="utf-8-sig"))
    state = snapshot(args.output, process_alive(launch["pid"]))
    if state["status"] in ("completed", "failed"):
        disabled = subprocess.run(["schtasks", "/Change", "/TN", launch["task_name"], "/Disable"],
                                  capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        state["monitor_disabled"] = disabled.returncode == 0
    text = json.dumps(state, ensure_ascii=False, indent=2)
    (args.output / "monitor.json").write_text(text, encoding="utf-8")
    with (args.output / "monitor_history.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(state, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
