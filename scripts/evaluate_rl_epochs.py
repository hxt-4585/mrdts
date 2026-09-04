"""Evaluate a checkpoint from train_rl_epochs.py."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    from methods.rl_baseline.epoch_training import evaluation_main
    evaluation_main()
