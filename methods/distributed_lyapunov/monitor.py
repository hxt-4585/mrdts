"""Read-only progress snapshot: python -m methods.distributed_lyapunov.monitor RUN."""
import argparse
import json
from pathlib import Path


def snapshot(run):
    result = {}
    for name in ['status.json','selection.json','pilot.json','pilot_confirmation.json']:
        path = run / name
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            result[name] = data[-1] if isinstance(data, list) else data
    for mode in ['joint','staged']:
        folder = run / mode
        status = folder / 'status.json'
        if status.exists():
            data = json.loads(status.read_text(encoding='utf-8'))
            history = folder / 'history.json'
            if history.exists():
                last = json.loads(history.read_text(encoding='utf-8'))[-1]
                data['last_episode'] = dict(phase=last['phase'], episode=last['episode'],
                    mean_delay_s=last['mean_delay_s'], max_queue_j=max(last['final_queue_j']),
                    worst_tail_slope=max(last['tail_slope_j_per_slot']),
                    empirically_stable=last['empirically_stable'], seconds=last['wall_seconds'])
            result[mode] = data
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.run), ensure_ascii=False, indent=2))
