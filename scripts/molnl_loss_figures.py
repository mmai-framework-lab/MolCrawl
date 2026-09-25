"""Drawing shared by the molecule_nat_lang loss figures.

Split out of scripts/molnl_bert_grid_curves.py when the GPT-2 ladder needed the same
axes, the same label placement and the same caption. What each caller keeps to itself is
what its own runs mean: which arms belong on one axis, what the floor is, and whether
there is a floor at all.
"""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#222222"
SIZE_COLOUR = {"small": "#1f6fb4", "medium": "#c8571b", "large": "#2e7d32",
               "ex-large": "#7b1fa2", "xl": "#7b1fa2"}
# For a figure whose arms are not one per size.
ARM_COLOURS = ("#c8571b", "#2e7d32", "#7b1fa2", "#1f6fb4", "#00838f", "#a1887f")


def new_figure():
    return plt.subplots(figsize=(8.4, 4.2))


def axes(ax, ylabel, xlabel="step"):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color="#dddddd", linewidth=.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def best_point(steps, vals):
    """(step, value) of the minimum -- what a label is anchored to."""
    i = min(range(len(vals)), key=lambda k: vals[k])
    return steps[i], vals[i]


def label_bests(ax, items, room=0.34, gap=0.062):
    """Mark each arm's minimum and name it beside the mark, without labels overlapping.

    The labels carry the arm's name, so none of these figures has a legend.

    Writing the value at the point does not work: arms in one figure reach their best at
    the same step and within a hair of each other, and the labels land on top of one
    another. Instead the axis is widened, every label is written in the margin that
    creates, and labels that would collide are pushed apart and joined to their point by
    a leader line. items is (step, value, text, colour).
    """
    lo, hi = ax.get_xlim()
    # The widening is margin, not schedule: a tick at 16,000 on a 12,000-step run reads
    # as if the run went there.
    ax.set_xlim(lo, hi + (hi - lo) * room)
    ax.set_xticks([t for t in ax.get_xticks() if lo <= t <= hi])
    ax.set_xlim(lo, hi + (hi - lo) * room)
    ax.figure.canvas.draw()   # a log axis has no usable transform before this

    to_axes = ax.transAxes.inverted()
    placed = []
    for x, y, text, colour in items:
        ax.plot([x], [y], marker="o", ms=5, color=colour, zorder=5)
        xf, yf = to_axes.transform(ax.transData.transform((x, y)))
        placed.append([xf, yf, yf, text, colour])

    # Push apart from the bottom up, then slide everything down if the stack overflows.
    placed.sort(key=lambda r: r[1])
    for i in range(1, len(placed)):
        placed[i][2] = max(placed[i][2], placed[i - 1][2] + gap)
    over = placed[-1][2] - 0.97 if placed and placed[-1][2] > 0.97 else 0
    for row in placed:
        row[2] = max(0.02, row[2] - over)

    label_x = 1 - room / (1 + room) + 0.015   # just inside the margin the widening made
    for xf, yf, y_label, text, colour in placed:
        ax.plot([xf + 0.006, label_x - 0.006], [yf, y_label], color=colour, linewidth=.6,
                alpha=.5, transform=ax.transAxes, zorder=4)
        ax.text(label_x, y_label, text, transform=ax.transAxes, va="center", fontsize=8,
                color=colour, zorder=5)


def fit_y(ax, series, pad=0.06):
    """Set the y-range from the curves themselves.

    Matplotlib's own limits are set by everything on the axes, so a reference line far
    from the data stretches the range until the curves are a flat band at one edge, and
    an annotation placed outside the data can push a curve past the frame. The range is
    taken from the drawn values instead, and a reference line is only drawn if it lands
    inside it (2026-09-25 order §8.2, §8.3).
    """
    values = [v for s_ in series for v in s_]
    lo, hi = min(values), max(values)
    if ax.get_yscale() == "log":
        import math
        span = math.log10(hi) - math.log10(lo)
        ax.set_ylim(10 ** (math.log10(lo) - span * pad), 10 ** (math.log10(hi) + span * pad))
    else:
        span = hi - lo or abs(hi) or 1.0
        ax.set_ylim(lo - span * pad, hi + span * pad)


def floor_line(ax, value, name):
    """Draw a reference floor when it is on the axis. Returns whether it was drawn."""
    lo, hi = ax.get_ylim()
    if lo <= value <= hi:
        ax.axhline(value, color="#888888", linestyle="--", linewidth=1)
        ax.annotate(f"{name} {value}", (0.015, value), xycoords=("axes fraction", "data"),
                    va="bottom", fontsize=8, color="#666666",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        return True
    return False


def save(fig, ax, title, note, path):
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    lines = [ln for raw in note.split("\n") for ln in textwrap.wrap(raw, 112)]
    fig.text(0.012, 0.012, "\n".join(lines), fontsize=7.5, color="#666666", va="bottom")
    fig.tight_layout(rect=(0, 0.035 + 0.026 * len(lines), 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
