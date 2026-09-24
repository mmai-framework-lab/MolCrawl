#!/usr/bin/env python3
"""Curves and the numbers behind them for the molecule_nat_lang BERT learning-rate grid.

Written for the 2026-09-24 order, which asks for the curve whenever a grid is run.

Two things it refuses to do, both from that order:

- it never draws a collapsed arm on the same axes as one that learned. The six that
  learned end between 0.07 and 0.19; the three that collapsed end between 7.4 and 9.3.
  On one linear axis the six are a flat line at the bottom.
- it never thins the evaluation points. Every one of the 120 per run is drawn.

The baseline is the unigram floor measured on this corpus (job 22503): a model that
predicts the corpus's token frequencies and reads nothing scores 3.8638 on the masked
positions. A loss above it means the run learned nothing usable.

    python scripts/molnl_bert_grid_curves.py --runs-root <dir> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASELINE = 3.8638          # unigram floor on the masked positions, mol_nl (job 22503)
SIZES = ("small", "medium", "large")
LRS = (("1e4", "1e-4", 1e-4), ("3e4", "3e-4", 3e-4), ("1e3", "1e-3", 1e-3))
LR_COLOUR = {"1e-4": "#1f6fb4", "3e-4": "#c8571b", "1e-3": "#2e7d32"}
SIZE_COLOUR = {"small": "#1f6fb4", "medium": "#c8571b", "large": "#2e7d32"}
# For the collapsed figure, where two of the three arms are the same size.
ARM_COLOURS = ("#c8571b", "#2e7d32", "#7b1fa2", "#1f6fb4", "#00838f", "#a1887f")
INK = "#222222"

# The five things a loss figure has to carry to be readable later.
PRECONDITIONS = ("max_steps 12,000 (96.57 epochs) · train 318,118 sequences, shuffled · "
                 "global batch 2,560 sequences · 31.46 G tokens seen · metric eval_loss_mask")


def read_run(run_dir):
    """(steps, values, segments) from the newest checkpoint's trainer_state.json."""
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[1]))
    if not cks:
        return [], [], {}
    hist = json.load(open(os.path.join(cks[-1], "trainer_state.json")))["log_history"]
    pts = sorted((e["step"], e["eval_loss_mask"]) for e in hist if "eval_loss_mask" in e)
    # Which segment produced each step. segments.log holds "<segment> <job id> <time>";
    # a run that never hit the time limit has one line and every point belongs to it.
    seg_path = os.path.join(run_dir, "segments.log")
    segments = {}
    if os.path.exists(seg_path):
        for line in open(seg_path):
            parts = line.split()
            if len(parts) >= 2:
                segments[int(parts[0])] = parts[1]
    return [s for s, _ in pts], [v for _, v in pts], segments


def collect(runs_root):
    runs = {}
    for size in SIZES:
        for tag, lr, _ in LRS:
            steps, vals, segs = read_run(os.path.join(runs_root, f"bert_{size}_lr{tag}"))
            if steps:
                runs[(size, lr)] = {"steps": steps, "vals": vals, "segments": segs}
    return runs


def learned(run):
    """Did the arm ever get below the unigram floor?"""
    return min(run["vals"]) < BASELINE


def _axes(ax, ylabel="eval_loss_mask"):
    ax.set_xlabel("step", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color="#dddddd", linewidth=.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def _best_point(steps, vals):
    """(step, value) of the minimum -- what a label is anchored to."""
    i = min(range(len(vals)), key=lambda k: vals[k])
    return steps[i], vals[i]


def _label_bests(ax, items, room=0.34, gap=0.062):
    """Mark each arm's minimum and name it beside the mark, without labels overlapping.

    The labels carry the arm's name, so none of these figures has a legend.

    Writing the value at the point does not work here: six arms reach their best at the
    same step, 11,900, within 0.13 of each other, so the labels land on top of one
    another. Instead the axis is widened, every label is written in the margin that
    creates, and labels that would collide are pushed apart and joined to their point by
    a leader line. items is (step, value, text, colour).
    """
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + (hi - lo) * room)
    # The widening is margin, not schedule: a tick at 16,000 on a 12,000-step run reads
    # as if the run went there.
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


def _baseline(ax, lo, hi):
    """Draw the floor when it is on the axis, and say so when it is not."""
    if lo <= BASELINE <= hi:
        ax.axhline(BASELINE, color="#888888", linestyle="--", linewidth=1)
        ax.annotate(f"unigram floor {BASELINE}", (0.015, BASELINE), xycoords=("axes fraction", "data"),
                    va="bottom", fontsize=8, color="#666666",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        return True
    return False


def _save(fig, ax, title, note, path):
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    lines = [ln for raw in note.split("\n") for ln in textwrap.wrap(raw, 112)]
    fig.text(0.012, 0.012, "\n".join(lines), fontsize=7.5, color="#666666", va="bottom")
    fig.tight_layout(rect=(0, 0.035 + 0.026 * len(lines), 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_by_size(runs, out_dir):
    """One figure per size, learning rates overlaid. Arms that learned only."""
    made = []
    for size in SIZES:
        arms = [(lr, runs[(size, lr)]) for _, lr, _ in LRS
                if (size, lr) in runs and learned(runs[(size, lr)])]
        if not arms:
            continue
        fig, ax = plt.subplots(figsize=(8.4, 4.2))
        for lr, run in arms:
            ax.plot(run["steps"], run["vals"], color=LR_COLOUR[lr], linewidth=1.3, label=f"lr {lr}")
        ax.set_yscale("log")
        _axes(ax)
        drawn = _baseline(ax, *ax.get_ylim())
        _label_bests(ax, [(*_best_point(r["steps"], r["vals"]),
                           f"lr {lr}: {min(r['vals']):.4f} @ {_best_point(r['steps'], r['vals'])[0]:,}",
                           LR_COLOUR[lr]) for lr, r in arms])
        note = (f"{PRECONDITIONS}\nlog scale, so the floor and the best values fit on one axis."
                + ("" if drawn else " The floor is off this axis."))
        made.append(_save(fig, ax, f"molecule_nat_lang BERT {size} - learning rates that learned",
                          note, os.path.join(out_dir, f"by-size-{size}.png")))
    return made


def fig_by_lr(runs, out_dir):
    """One figure per learning rate, sizes overlaid. Arms that learned only."""
    made = []
    for _, lr, _ in LRS:
        arms = [(size, runs[(size, lr)]) for size in SIZES
                if (size, lr) in runs and learned(runs[(size, lr)])]
        if not arms:
            continue
        fig, ax = plt.subplots(figsize=(8.4, 4.2))
        for size, run in arms:
            ax.plot(run["steps"], run["vals"], color=SIZE_COLOUR[size], linewidth=1.3, label=size)
        ax.set_yscale("log")
        _axes(ax)
        drawn = _baseline(ax, *ax.get_ylim())
        _label_bests(ax, [(*_best_point(r["steps"], r["vals"]),
                           f"{size}: {min(r['vals']):.4f} @ {_best_point(r['steps'], r['vals'])[0]:,}",
                           SIZE_COLOUR[size]) for size, r in arms])
        note = (f"{PRECONDITIONS}\nSizes that collapsed at this learning rate are not drawn here; "
                "see the collapsed figure." + ("" if drawn else " The floor is off this axis."))
        made.append(_save(fig, ax, f"molecule_nat_lang BERT lr {lr} - sizes that learned",
                          note, os.path.join(out_dir, f"by-lr-{lr}.png")))
    return made


def fig_collapsed(runs, out_dir):
    """The arms that never got below the floor, on their own axis."""
    arms = [(size, lr, runs[(size, lr)]) for size in SIZES for _, lr, _ in LRS
            if (size, lr) in runs and not learned(runs[(size, lr)])]
    if not arms:
        return []
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for (size, lr, run), colour in zip(arms, ARM_COLOURS):
        ax.plot(run["steps"], run["vals"], color=colour, linewidth=1.3, label=f"{size} lr {lr}")
    _axes(ax)
    ax.set_ylim(min(BASELINE * 0.97, ax.get_ylim()[0]), ax.get_ylim()[1])
    _baseline(ax, *ax.get_ylim())
    _label_bests(ax, [(*_best_point(r["steps"], r["vals"]),
                       f"{size} lr {lr}: {min(r['vals']):.4f} @ {_best_point(r['steps'], r['vals'])[0]:,}", c)
                      for (size, lr, r), c in zip(arms, ARM_COLOURS)], room=0.34)
    note = (f"{PRECONDITIONS}\nEvery point of every one of these is above the floor: their best is "
            "worse than predicting token frequencies. Marked points are each arm's minimum.")
    return [_save(fig, ax, "molecule_nat_lang BERT - the arms that collapsed", note,
                  os.path.join(out_dir, "collapsed.png"))]


def fig_tail(runs, out_dir, frac=0.20):
    """The last 20% of the schedule, for the arms that learned: is the best at the end?"""
    arms = [(size, lr, runs[(size, lr)]) for size in SIZES for _, lr, _ in LRS
            if (size, lr) in runs and learned(runs[(size, lr)])]
    if not arms:
        return []
    cut = max(max(r["steps"]) for _, _, r in arms) * (1 - frac)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    items = []
    for size, lr, run in arms:
        pts = [(s, v) for s, v in zip(run["steps"], run["vals"]) if s >= cut]
        ax.plot([s for s, _ in pts], [v for _, v in pts], color=SIZE_COLOUR[size],
                linestyle={"1e-4": "-", "3e-4": "--", "1e-3": ":"}[lr],
                linewidth=1.3, label=f"{size} lr {lr}")
        bx, by = _best_point([s for s, _ in pts], [v for _, v in pts])
        items.append((bx, by, f"{size} lr {lr}: {by:.4f} @ {bx:,}", SIZE_COLOUR[size]))
    _axes(ax)
    _label_bests(ax, items, room=0.46)
    note = (f"{PRECONDITIONS}\nLast {int(frac * 100)}% of the schedule, y near the best values. "
            f"The floor {BASELINE} is far above this axis. Whether the minimum sits at the end "
            "decides whether the step budget is the thing to change.")
    return [_save(fig, ax, f"molecule_nat_lang BERT - last {int(frac * 100)}% of the schedule",
                  note, os.path.join(out_dir, "tail-last20pct.png"))]


def write_tsv(runs, out_dir):
    path = os.path.join(out_dir, "molnl-bert-grid-eval-loss-mask.tsv")
    with open(path, "w") as fh:
        fh.write(f"# molecule_nat_lang BERT learning-rate grid, eval_loss_mask at every evaluation\n")
        fh.write(f"# {PRECONDITIONS}\n")
        fh.write(f"# unigram floor on this corpus: {BASELINE} (job 22503)\n")
        fh.write("size\tlearning_rate\tstep\teval_loss_mask\tsegment\n")
        for size in SIZES:
            for _, lr, _ in LRS:
                run = runs.get((size, lr))
                if not run:
                    continue
                segment = min(run["segments"]) if run["segments"] else 1
                for s, v in zip(run["steps"], run["vals"]):
                    fh.write(f"{size}\t{lr}\t{s}\t{v:.6f}\t{segment}\n")
    return path


def wobble(runs, n=10):
    """Spread of the last n evaluation points -- a stand-in for a repeat measurement.

    There is no second run at the same settings, so the run-to-run spread is unmeasured.
    What this shows is how much the number moves between neighbouring evaluations once a
    run has converged: each evaluation draws its masked positions afresh, so a difference
    smaller than this is not a difference.
    """
    rows = []
    for size in SIZES:
        for _, lr, _ in LRS:
            run = runs.get((size, lr))
            if run and learned(run):
                tail = run["vals"][-n:]
                rows.append((size, lr, min(run["vals"]), st.pstdev(tail), max(tail) - min(tail)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True, help="directory holding bert_<size>_lr<tag>/")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    runs = collect(a.runs_root)
    if not runs:
        print(f"no runs under {a.runs_root}")
        return 1
    made = (fig_by_size(runs, a.out_dir) + fig_by_lr(runs, a.out_dir)
            + fig_collapsed(runs, a.out_dir) + fig_tail(runs, a.out_dir))
    tsv = write_tsv(runs, a.out_dir)
    print(f"runs read: {len(runs)}")
    for p in made + [tsv]:
        print(f"  {os.path.basename(p)}  {os.path.getsize(p):,} bytes")
    print("\nspread of the last 10 evaluations (no repeat run exists; this is not run-to-run spread)")
    print(f"  {'size':<8}{'lr':>6}{'best':>9}{'sd(last10)':>12}{'range(last10)':>15}")
    for size, lr, best, sd, rng in wobble(runs):
        print(f"  {size:<8}{lr:>6}{best:>9.4f}{sd:>12.4f}{rng:>15.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
