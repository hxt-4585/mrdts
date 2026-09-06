"""正式评估入口：python -m experiments.run 或 PyCharm 直接运行本文件。"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.cli import parse_config
from experiments.runner import evaluate


def main(argv=None):
    parser, _, config = parse_config(argv)
    try:
        output = evaluate(config)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Evaluation results: {output}")
    return output


if __name__ == "__main__":
    main()
