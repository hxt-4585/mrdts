"""绘制单次正式实验的逐时隙指标，使用不依赖桌面窗口的后端。"""

import argparse
import csv
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.config import project_path


def plot_run(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = project_path(directory)
    with (directory / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No slot metrics to plot")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for episode in sorted({int(row["episode"]) for row in rows}):
        selected = [row for row in rows if int(row["episode"]) == episode]
        slots = [int(row["slot"]) for row in selected]
        axes[0].plot(slots, [float(row["failure_rate"]) for row in selected], label=f"Episode {episode}")
        axes[1].plot(slots, [float(row["truncated_delay_sum_s"]) / int(row["dag_count"])
                            for row in selected], label=f"Episode {episode}")
    axes[0].set(ylabel="DAG failure rate", ylim=(-0.02, 1.02))
    axes[1].set(ylabel="Mean truncated delay (s)")
    for axis in axes:
        axis.set_xlabel("Slot")
        axis.legend()
    output = directory / "figures" / "metrics.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    args = parser.parse_args()
    print(plot_run(args.run_directory))
