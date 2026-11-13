"""Evaluate a frozen HEFT flight checkpoint on held-out workload epochs."""
import argparse
from pathlib import Path

from .run import evaluate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--epochs', nargs='+', type=int)
    parser.add_argument('--slots', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.epochs, args.slots)
    write_json(args.output or args.checkpoint.parent / 'evaluation.json', report)
    print(report['summary'])


if __name__ == '__main__':
    main()
