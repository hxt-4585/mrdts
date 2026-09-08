"""Export source-backed offline figures from a training run."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from itertools import groupby
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
    from .style import stage_colors

    colours = dict(stage_colors(rows))
    for stage, segment in groupby(rows, key=lambda row: row.stage):
        colour = colours[stage]
        selected = [row for row in segment if getattr(row, field) is not None]
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

    panels = [("mean_reward", "Mean step reward", "Mean reward"),
              ("epoch_return", "Epoch return", "Sum of rewards")]
    if any(row.mean_delay_s is not None for row in data.epochs + data.validation):
        panels.append(("mean_delay_s", "Mean delay", "Seconds"))
    figure, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 3.5),
                                layout="constrained")
    for axis, (field, title, ylabel) in zip(axes, panels):
        _plot_stage_series(axis, data.epochs, field)
        validation = [row for row in data.validation if getattr(row, field) is not None]
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
    figure.suptitle(f"Training metrics | seed {data.seed if data.seed is not None else 'unknown'}")
    return figure


def update_fields(data: RunData, *, diagnostics=False):
    return list(dict.fromkeys(name for row in data.updates for name in row.metrics
                              if (name in {"approx_kl", "entropy"}) == diagnostics))


def draw_update_panels(figure, data: RunData, *, diagnostics=False, smooth=True):
    """Rebuild panels so a live monitor can discover newly logged metrics."""
    from .style import stage_colors, finish_axis

    figure.clear()
    fields = update_fields(data, diagnostics=diagnostics)
    axes = figure.subplots(1, max(1, len(fields)), squeeze=False)[0]
    if not fields:
        axes[0].set_axis_off()
        axes[0].text(.5, .5, "No diagnostic data" if diagnostics else "Waiting for loss data",
                     ha="center", va="center", transform=axes[0].transAxes)
        return
    for axis, field in zip(axes, fields):
        title = {"approx_kl": "Approximate KL", "entropy": "Policy entropy"}.get(
            field, field.replace("_", " ").capitalize())
        colours = dict(stage_colors(data.epochs + data.updates))
        for stage, segment in groupby(data.updates, key=lambda row: row.stage):
            colour = colours[stage]
            rows = [row for row in segment if field in row.metrics]
            if rows:
                x = [row.update for row in rows]
                values = [row.metrics[field] for row in rows]
                axis.plot(x, values, color=colour, alpha=.35, marker=".", label=f"{stage} raw")
                if smooth and not diagnostics and len(rows) >= 5:
                    axis.plot(x[4:], stage_smooth([row.stage for row in rows], values)[4:], color=colour,
                              marker="." if len(rows) == 5 else None,
                              linewidth=1.7, label=f"{stage} smoothed")
                if diagnostics:
                    stopped = [row for row in rows if row.early_stop]
                    if stopped:
                        axis.scatter([row.update for row in stopped],
                                     [row.metrics[field] for row in stopped], facecolors="none",
                                     edgecolors=colour, s=45, label=f"{stage} early stop")
        if field == "value_loss":
            axis.set_yscale("symlog", linthresh=1e-4)
            title += " (symlog)"
        axis.set_title(title)
        axis.set_ylabel(title)
        finish_axis(axis, "Update")
        axis.legend(fontsize=7)
    mark_stage_switches(axes, data.updates, "update")


def losses_figure(data: RunData):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(4 * max(1, len(update_fields(data))), 3.4), layout="constrained")
    draw_update_panels(figure, data)
    figure.suptitle("Optimization losses")
    return figure


def diagnostics_figure(data: RunData):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(4 * max(1, len(update_fields(data, diagnostics=True))), 3.4),
                         layout="constrained")
    draw_update_panels(figure, data, diagnostics=True, smooth=False)
    figure.suptitle("Update diagnostics")
    return figure


def _write_source(data: RunData, output: Path):
    rows = []
    for record_type, records in (("epoch", data.epochs), ("update", data.updates),
                                 ("validation", data.validation)):
        for record in records:
            values = asdict(record)
            values.update(values.pop("metrics", {}))
            rows.append({"record_type": record_type, "seed": data.seed, "mode": data.mode,
                         "status": data.status, **values})
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
    figures = {"epoch-reward": reward_figure(data)}
    if update_fields(data):
        figures["losses"] = losses_figure(data)
    if update_fields(data, diagnostics=True):
        figures["diagnostics"] = diagnostics_figure(data)
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
