"""Shared plotting style for training figures."""

MEMBER = "#4477AA"
MASTER = "#EE6677"
VALIDATION = "#228833"
NEUTRAL = "#666666"


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
