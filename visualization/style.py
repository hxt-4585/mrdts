"""Shared plotting style for training figures."""

STAGE_COLORS = ("#4477AA", "#EE6677", "#AA3377", "#CCBB44", "#66CCEE", "#BBBBBB")
VALIDATION = "#228833"
NEUTRAL = "#666666"


def stage_colors(rows):
    """Assign colours by first appearance, without knowing algorithm stage names."""
    return [(stage, STAGE_COLORS[index % len(STAGE_COLORS)])
            for index, stage in enumerate(dict.fromkeys(row.stage for row in rows))]


def apply_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.spines.right": False, "axes.spines.top": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def finish_axis(axis, xlabel):
    axis.set_xlabel(xlabel)
    axis.grid(axis="y", linewidth=.5, alpha=.2)
