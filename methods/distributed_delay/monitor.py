"""One-shot monitor invoked periodically by Windows Task Scheduler."""

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess


def inspect(run, task_name=None):
    metadata = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))['training']
    processes = subprocess.run(['tasklist.exe', '/FI', f'PID eq {metadata["pid"]}', '/FO', 'CSV', '/NH'],
                               capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    alive = any(len(row) > 1 and row[1] == str(metadata['pid'])
                for row in csv.reader(io.StringIO(processes.stdout)))
    snapshot = dict(checked_at=datetime.now(timezone.utc).isoformat(), process_alive=alive,
                    status=metadata['status'], phase=metadata['phase'], episode=metadata['episode'],
                    episodes_planned=config['episodes'], best_validation_delay_s=metadata.get('best_validation_delay_s'),
                    comparison=metadata.get('comparison'))
    history = run / 'episodes.csv'
    if history.exists():
        with history.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            snapshot['last_episode'] = rows[-1]
        if len(rows) >= 2:
            window = rows[-6:]
            per_episode = ((float(window[-1]['elapsed_s']) - float(window[0]['elapsed_s'])) /
                           (int(window[-1]['episode']) - int(window[0]['episode'])))
            snapshot['estimated_training_minutes_remaining'] = (
                config['episodes'] - int(rows[-1]['episode'])) * per_episode / 60.
    terminal = metadata['status'] in ('completed', 'failed')
    if not terminal and not alive:
        snapshot['attention'] = 'Training exited without a terminal status; inspect stderr.log.'
    if terminal or not alive:
        (run / 'completion.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
        if task_name is not None:
            disabled = subprocess.run(['schtasks.exe', '/Change', '/TN', task_name, '/DISABLE'],
                                      capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            snapshot['task_disabled'] = disabled.returncode == 0
    temporary = run / 'monitor.tmp'
    temporary.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    temporary.replace(run / 'monitor.json')
    with (run / 'monitor_history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(snapshot) + '\n')
    return snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--task-name')
    args = parser.parse_args()
    inspect(args.run, args.task_name)
