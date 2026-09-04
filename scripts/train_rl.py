"""Train the optional baseline: python scripts/train_rl.py --help."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    try:
        from methods.rl_baseline.training import main
    except ModuleNotFoundError as error:
        if error.name == 'torch':
            raise SystemExit('Install RL dependencies: uv pip install --python .venv/Scripts/python.exe -r requirements-rl.txt')
        raise
    main()
