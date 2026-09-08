"""Export source-backed offline figures from a PPO training run."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from .readers import RunData, project_path, read_run


def stage_smooth(stages, values, window=5):
    """Trailing mean that starts over at each stage boundary."""
    result = []
    stage_values = []
    previous = None
    for stage, value in zip(stages, values):
        if stage != previous:
            stage_values = []
        stage_values.append(value)
        result.append(sum(stage_values[-window:]) / len(stage_values[-window:]))
        previous = stage
    return result


def _plot_stage_series(axis, rows, field, *, label_prefix="Training"):
    from .style import MASTER, MEMBER

    for stage, colour in (("member", MEMBER), ("master", MASTER)):
        selected = [row for row in rows if row.stage == stage]
        if not selected:
            continue
        x = [row.epoch for row in selected]
        values = [getattr(row, field) for row in selected]
        axis.plot(x, values, color=colour, alpha=.3, marker=".", linewidth=.8,
                  label=f"{label_prefix} {stage} raw")
        if len(selected) >= 5:
            smooth = stage_smooth([row.stage for row in selected], values)
            axis.plot(x[4:], smooth[4:], color=colour, linewidth=1.7,
                      marker="." if len(selected) == 5 else None,
                      label=f"{label_prefix} {stage} smoothed")


def mark_stage_switches(axes, rows, x_field):
    """Mark stage transitions at the midpoint between adjacent log keys."""
    for previous, current in zip(rows, rows[1:]):
        if previous.stage != current.stage:
            boundary = (getattr(previous, x_field) + getattr(current, x_field)) / 2
            for axis in axes:
                axis.axvline(boundary, color="#777777", linestyle=":", linewidth=1)


def reward_figure(data: RunData):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from .style import VALIDATION, finish_axis

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    for axis, field, title, ylabel in zip(
            axes, ("mean_reward", "epoch_return", "mean_delay_s"),
            ("Mean step reward", "Epoch return", "Mean censored delay"),
            ("Mean reward", "Sum of rewards", "Seconds")):
        _plot_stage_series(axis, data.epochs, field)
        validation = data.validation
        if field == "epoch_return":
            training_steps = {(row.epoch, row.steps) for row in data.epochs}
            validation = [row for row in validation if (row.epoch, row.steps) in training_steps]
        if validation:
            axis.plot([row.epoch for row in validation],
                      [getattr(row, field) for row in validation], color=VALIDATION,
                      marker="o", markersize=3.5, linestyle="--", label="Validation")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        finish_axis(axis, "Epoch")
    mark_stage_switches(axes, data.epochs, "epoch")
    axes[0].legend(fontsize=7)
    figure.suptitle(f"Training reward and delay | seed {data.seed if data.seed is not None else 'unknown'}")
    return figure


def losses_figure(data: RunData):
    import matplotlib.pyplot as plt
    from .style import MASTER, MEMBER, finish_axis

    figure, axes = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
    for axis, field, title in zip(axes, ("actor_loss", "value_loss"),
                                  ("Actor loss", "Value loss")):
        for stage, colour in (("member", MEMBER), ("master", MASTER)):
            rows = [row for row in data.updates if row.stage == stage]
            if rows:
                x = [row.update for row in rows]
                values = [getattr(row, field) for row in rows]
                axis.plot(x, values, color=colour, alpha=.35, marker=".", label=f"{stage} raw")
                if len(rows) >= 5:
                    axis.plot(x[4:], stage_smooth([row.stage for row in rows], values)[4:], color=colour,
                              marker="." if len(rows) == 5 else None,
                              linewidth=1.7, label=f"{stage} smoothed")
        if field == "value_loss":
            axis.set_yscale("symlog", linthresh=1e-4)
            title += " (symlog)"
        axis.set_title(title)
        axis.set_ylabel(field.replace("_", " ").title())
        finish_axis(axis, "Update")
    mark_stage_switches(axes, data.updates, "update")
    axes[0].legend(fontsize=7)
    figure.suptitle("PPO optimization losses")
    return figure


def diagnostics_figure(data: RunData):
    import matplotlib.pyplot as plt
    from .style import MASTER, MEMBER, finish_axis

    figure, axes = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
    for axis, field, title in zip(axes, ("approx_kl", "entropy"),
                                  ("Approximate KL", "Policy entropy")):
        for stage, colour in (("member", MEMBER), ("master", MASTER)):
            rows = [row for row in data.updates if row.stage == stage]
            if rows:
                axis.plot([row.update for row in rows], [getattr(row, field) for row in rows],
                          color=colour, marker=".", label=stage)
                stopped = [row for row in rows if row.early_stop]
                if stopped:
                    axis.scatter([row.update for row in stopped],
                                 [getattr(row, field) for row in stopped], facecolors="none",
                                 edgecolors=colour, s=45, label=f"{stage} early stop")
        axis.set_title(title)
        axis.set_ylabel(title)
        finish_axis(axis, "Update")
    axes[0].legend(fontsize=7)
    figure.suptitle("PPO update diagnostics")
    return figure


def _write_source(data: RunData, output: Path):
    rows = []
    for record_type, records in (("epoch", data.epochs), ("update", data.updates),
                                 ("validation", data.validation)):
        for record in records:
            rows.append({"record_type": record_type, "seed": data.seed, "mode": data.mode,
                         "status": data.status, **asdict(record)})
    names = ["record_type", "seed", "mode", "status"] + sorted(
        {key for row in rows for key in row} - {"record_type", "seed", "mode", "status"})
    with (output / "source.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export(run, output=None):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from .style import apply_style

    data = read_run(run)
    apply_style()
    output = project_path(output) if output is not None else data.run / "figures" / "training"
    output.mkdir(parents=True, exist_ok=True)
    figures = {"epoch-reward": reward_figure(data), "losses": losses_figure(data),
               "diagnostics": diagnostics_figure(data)}
    for stem, figure in figures.items():
        for suffix in ("png", "svg", "pdf"):
            figure.savefig(output / f"{stem}.{suffix}", dpi=250, bbox_inches="tight")
        plt.close(figure)
    _write_source(data, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Training run directory")
    parser.add_argument("--output", help="Output directory (default: RUN/figures/training)")
    args = parser.parse_args(argv)
    try:
        print(export(args.run, args.output))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
