"""Frozen policy evaluation without replay insertion or parameter updates."""
import argparse
from pathlib import Path

from methods.mrdts_delay.run import evaluate
from methods.mrdts_delay.storage import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--epochs', type=int, nargs='+')
    parser.add_argument('--slots', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.epochs, slots=args.slots, workers=args.workers)
    output = args.output or args.checkpoint.parent / 'evaluation.json'
    write_json(output, result)
    print(result['summary'])


if __name__ == '__main__':
    main()
