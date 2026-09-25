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

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from molnl_loss_figures import (ARM_COLOURS, SIZE_COLOUR, axes, best_point,  # noqa: E402
                                fit_y, floor_line, label_bests, new_figure, save)

BASELINE = 3.8638          # unigram floor on the masked positions, mol_nl (job 22503)
SIZES = ("small", "medium", "large")
LRS = (("1e4", "1e-4", 1e-4), ("3e4", "3e-4", 3e-4), ("1e3", "1e-3", 1e-3))
LR_COLOUR = {"1e-4": "#1f6fb4", "3e-4": "#c8571b", "1e-3": "#2e7d32"}
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


def fig_by_size(runs, out_dir):
    """One figure per size, learning rates overlaid. Arms that learned only."""
    made = []
    for size in SIZES:
        arms = [(lr, runs[(size, lr)]) for _, lr, _ in LRS
                if (size, lr) in runs and learned(runs[(size, lr)])]
        if not arms:
            continue
        fig, ax = new_figure()
        for lr, run in arms:
            ax.plot(run["steps"], run["vals"], color=LR_COLOUR[lr], linewidth=1.3, label=f"lr {lr}")
        ax.set_yscale("log")
        axes(ax, "eval_loss_mask")
        fit_y(ax, [r["vals"] for _, r in arms])
        drawn = floor_line(ax, BASELINE, "unigram floor")
        label_bests(ax, [(*best_point(r["steps"], r["vals"]),
                           f"lr {lr}: {min(r['vals']):.4f} @ {best_point(r['steps'], r['vals'])[0]:,}",
                           LR_COLOUR[lr]) for lr, r in arms])
        note = (f"{PRECONDITIONS}\nlog scale, so the floor and the best values fit on one axis."
                + ("" if drawn else " The floor is off this axis."))
        made.append(save(fig, ax, f"molecule_nat_lang BERT {size} - learning rates that learned",
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
        fig, ax = new_figure()
        for size, run in arms:
            ax.plot(run["steps"], run["vals"], color=SIZE_COLOUR[size], linewidth=1.3, label=size)
        ax.set_yscale("log")
        axes(ax, "eval_loss_mask")
        fit_y(ax, [r["vals"] for _, r in arms])
        drawn = floor_line(ax, BASELINE, "unigram floor")
        label_bests(ax, [(*best_point(r["steps"], r["vals"]),
                           f"{size}: {min(r['vals']):.4f} @ {best_point(r['steps'], r['vals'])[0]:,}",
                           SIZE_COLOUR[size]) for size, r in arms])
        note = (f"{PRECONDITIONS}\nSizes that collapsed at this learning rate are not drawn here; "
                "see the collapsed figure." + ("" if drawn else " The floor is off this axis."))
        made.append(save(fig, ax, f"molecule_nat_lang BERT lr {lr} - sizes that learned",
                          note, os.path.join(out_dir, f"by-lr-{lr}.png")))
    return made


def fig_collapsed(runs, out_dir):
    """The arms that never got below the floor, on their own axis."""
    arms = [(size, lr, runs[(size, lr)]) for size in SIZES for _, lr, _ in LRS
            if (size, lr) in runs and not learned(runs[(size, lr)])]
    if not arms:
        return []
    fig, ax = new_figure()
    for (size, lr, run), colour in zip(arms, ARM_COLOURS):
        ax.plot(run["steps"], run["vals"], color=colour, linewidth=1.3, label=f"{size} lr {lr}")
    axes(ax, "eval_loss_mask")
    fit_y(ax, [r["vals"] for _, _, r in arms])
    # The floor is the point of this figure -- every one of these stayed above it -- so
    # here the range is widened to hold it rather than the line being dropped.
    ax.set_ylim(min(BASELINE * 0.97, ax.get_ylim()[0]), ax.get_ylim()[1])
    floor_line(ax, BASELINE, "unigram floor")
    label_bests(ax, [(*best_point(r["steps"], r["vals"]),
                       f"{size} lr {lr}: {min(r['vals']):.4f} @ {best_point(r['steps'], r['vals'])[0]:,}", c)
                      for (size, lr, r), c in zip(arms, ARM_COLOURS)], room=0.34)
    note = (f"{PRECONDITIONS}\nEvery point of every one of these is above the floor: their best is "
            "worse than predicting token frequencies. Marked points are each arm's minimum.")
    return [save(fig, ax, "molecule_nat_lang BERT - the arms that collapsed", note,
                  os.path.join(out_dir, "collapsed.png"))]


def fig_tail(runs, out_dir, frac=0.20):
    """The last 20% of the schedule, for the arms that learned: is the best at the end?"""
    arms = [(size, lr, runs[(size, lr)]) for size in SIZES for _, lr, _ in LRS
            if (size, lr) in runs and learned(runs[(size, lr)])]
    if not arms:
        return []
    cut = max(max(r["steps"]) for _, _, r in arms) * (1 - frac)
    fig, ax = new_figure()
    items, elsewhere, drawn = [], [], []
    for size, lr, run in arms:
        pts = [(s, v) for s, v in zip(run["steps"], run["vals"]) if s >= cut]
        ax.plot([s for s, _ in pts], [v for _, v in pts], color=SIZE_COLOUR[size],
                linestyle={"1e-4": "-", "3e-4": "--", "1e-3": ":"}[lr],
                linewidth=1.3, label=f"{size} lr {lr}")
        drawn.append([v for _, v in pts])
        # The mark means one thing in every figure: the run's own best. When that falls
        # before this window it is not drawn here, and the caption says where it is.
        bx, by = best_point(run["steps"], run["vals"])
        if bx >= cut:
            items.append((bx, by, f"{size} lr {lr}: {by:.4f} @ {bx:,}", SIZE_COLOUR[size]))
        else:
            elsewhere.append(f"{size} lr {lr} {by:.4f} at {bx:,}")
    axes(ax, "eval_loss_mask")
    fit_y(ax, drawn)
    label_bests(ax, items, room=0.46)
    note = (f"{PRECONDITIONS}\nLast {int(frac * 100)}% of the schedule, y near the best values. "
            f"The floor {BASELINE} is far above this axis. Marks are each run's own best."
            + (" Best before this window, so not marked here: " + "; ".join(elsewhere) + "."
               if elsewhere else "")
            + " Whether the minimum sits at the end decides whether the step budget is the "
            "thing to change.")
    return [save(fig, ax, f"molecule_nat_lang BERT - last {int(frac * 100)}% of the schedule",
                  note, os.path.join(out_dir, "tail-last20pct.png"))]


def write_tsv(runs, out_dir):
    path = os.path.join(out_dir, "molnl-bert-grid-eval-loss-mask.tsv")
    with open(path, "w") as fh:
        fh.write("# molecule_nat_lang BERT learning-rate grid, eval_loss_mask at every evaluation\n")
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
