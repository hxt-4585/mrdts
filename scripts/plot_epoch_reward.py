"""Plot reward against complete environment epochs or explicitly labelled legacy rounds."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    from methods.rl_baseline.epoch_reward_plot import main
    main()
