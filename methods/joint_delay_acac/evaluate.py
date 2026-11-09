"""Re-evaluate a joint checkpoint on frozen held-out task streams."""
import argparse
from pathlib import Path

from methods.joint_delay_acac.run import evaluate
from methods.joint_delay_acac.support import write_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--epoch-start', type=int, default=6000000)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--slots', type=int, default=500)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = evaluate(args.checkpoint, list(range(args.epoch_start, args.epoch_start + args.episodes)),
                    slots=args.slots, workers=args.workers)
    destination = args.output or args.checkpoint.parent / 'evaluation.json'
    write_json(destination, dict(episodes=rows, mean_delay_s=sum(r['mean_delay_s'] for r in rows) / len(rows)))
    print(destination)

