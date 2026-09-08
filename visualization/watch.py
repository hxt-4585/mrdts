"""Independent live reward/loss monitor for training logs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .readers import project_path, read_run


def _draw_reward(figure, axes, data):
    from .style import VALIDATION, finish_axis
    from .training import _plot_stage_series, mark_stage_switches

    for axis in axes:
        axis.clear()
    for axis, field, title in zip(axes, ("mean_reward", "epoch_return"),
                                  ("Mean step reward", "Epoch return")):
        _plot_stage_series(axis, data.epochs, field)
        validation = data.validation
        if field == "epoch_return":
            training_steps = {(row.epoch, row.steps) for row in data.epochs}
            validation = [row for row in validation if (row.epoch, row.steps) in training_steps]
        if validation:
            axis.plot([row.epoch for row in validation],
                      [getattr(row, field) for row in validation], color=VALIDATION,
                      marker="o", linestyle="--", label="Validation")
        axis.set_title(title)
        finish_axis(axis, "Epoch")
    mark_stage_switches(axes, data.epochs, "epoch")
    if data.epochs:
        last = data.epochs[-1]
        status = f"Epoch {last.epoch} | {last.stage} | mean reward {last.mean_reward:.5g}"
        axes[0].legend(fontsize=7)
    else:
        status = "Waiting for the first completed epoch"
    figure.suptitle(status)
    figure.canvas.draw_idle()
    return status


def _draw_losses(figure, axes, data):
    from .training import draw_update_panels

    draw_update_panels(figure, data, smooth=False)
    figure.suptitle("Optimization losses" if data.updates else "Waiting for the first completed update")
    figure.canvas.draw_idle()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Training run directory")
    parser.add_argument("--once", action="store_true", help="Read once and export headless PNGs")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    run = project_path(args.run)
    import matplotlib
    matplotlib.use("Agg" if args.once else "TkAgg", force=True)
    import matplotlib.pyplot as plt
    from .style import apply_style
    apply_style()
    reward_figure, reward_axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    loss_figure, loss_axes = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
    output = run / "figures" / "training"
    previous = None

    def refresh():
        nonlocal previous
        try:
            data = read_run(run, allow_pending=True)
            if data != previous:
                print(_draw_reward(reward_figure, reward_axes, data), flush=True)
                _draw_losses(loss_figure, loss_axes, data)
                if args.once or data.epochs or data.updates:
                    output.mkdir(parents=True, exist_ok=True)
                    reward_figure.savefig(output / "live-reward.png", dpi=150)
                    loss_figure.savefig(output / "live-losses.png", dpi=150)
                previous = data
        except (OSError, ValueError) as error:
            if args.once:
                raise
            print(f"Log read error; retrying: {error}", flush=True)

    refresh()
    if args.once:
        plt.close(reward_figure)
        plt.close(loss_figure)
        return
    reward_figure.canvas.manager.set_window_title(f"Training reward - {run}")
    loss_figure.canvas.manager.set_window_title(f"Training losses - {run}")
    timers = []
    for figure in (reward_figure, loss_figure):
        timer = figure.canvas.new_timer(interval=max(1, int(args.interval * 1000)))
        timer.add_callback(refresh)
        timer.start()
        timers.append(timer)
    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close("all")
    finally:
        for timer in timers:
            timer.stop()


if __name__ == "__main__":
    main()
