#!/usr/bin/env python3
"""The molecule_nat_lang GPT-2 learning-rate grid, compared over a window.

The order that set this grid up says to compare the arms over the mean of a window of
evaluations, not by their best points: a single evaluation moves by about 0.01, and the
minimum of a noisy series is biased downward by the act of picking it -- the more
evaluations a run has, the lower its best tends to look for no better reason.

So the comparison here is the mean of the last N evaluations of the schedule, the same
steps for every arm, which is only meaningful because every arm runs the identical
schedule: 1,500 iterations, evaluated every 50. The best point is printed beside it to
show how far the two disagree.

    python scripts/molnl_gpt2_grid_window.py --runs-root <gpt2-output-grid> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from molnl_loss_figures import (SIZE_COLOUR, axes, best_point, fit_y,  # noqa: E402
                                floor_line, label_bests, new_figure, save)

UNIGRAM_FLOOR = 4.6514      # next-token, this corpus (scripts/molnl_unigram_floor.py, job 138304)
GLOBAL_BATCH, BLOCK, TRAIN_BLOCKS = 2560, 1024, 318118
SIZE_ORDER = ("small", "medium", "large", "xl")
# gpt2_<size>_1500_lr<tag>: 1p2e3 is 1.2e-3, 6e4 is 6e-4.
NAME = re.compile(r"gpt2_(?P<size>[a-z]+)_(?P<iters>\d+)_lr(?P<tag>[0-9p]+e\d)$")


def rate_of(tag):
    mantissa, exponent = tag.split("e")
    return float(mantissa.replace("p", ".")) * 10 ** -int(exponent)


def read_run(run_dir):
    best = []
    for path in glob.glob(os.path.join(run_dir, "logging_*.csv")):
        rows = [r for r in csv.reader(open(path)) if r and r[0].strip().isdigit()]
        if len(rows) > len(best):
            best = rows
    return ([int(r[0]) for r in best], [float(r[2]) for r in best], [float(r[1]) for r in best])


def collect(runs_root, iters):
    runs = {}
    for d in sorted(glob.glob(os.path.join(runs_root, "gpt2_*_lr*"))):
        m = NAME.match(os.path.basename(d))
        if not m or int(m["iters"]) != iters:
            continue
        steps, val, train = read_run(d)
        if steps:
            runs[(m["size"], rate_of(m["tag"]))] = {
                "steps": steps, "val": val, "train": train,
                "complete": max(steps) >= iters, "name": os.path.basename(d),
            }
    return runs


def window(run, last_n, iters, every):
    """Mean of the last ``last_n`` evaluations, and the window's first step."""
    start = iters - (last_n - 1) * every
    pts = [v for s, v in zip(run["steps"], run["val"]) if s >= start]
    return (st.mean(pts) if pts else None), start, len(pts)


def window_sd(run, last_n, iters, every):
    """How much the arm moves inside its own window -- the scale a gap is read against."""
    start = iters - (last_n - 1) * every
    pts = [v for s, v in zip(run["steps"], run["val"]) if s >= start]
    return st.pstdev(pts) if len(pts) > 1 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--last", type=int, default=10, help="evaluations in the window")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    runs = collect(a.runs_root, a.iters)
    if not runs:
        print(f"no runs under {a.runs_root}")
        return 1
    done = {k: r for k, r in runs.items() if r["complete"]}
    tokens = a.iters * GLOBAL_BATCH * BLOCK
    pre = (f"max_iters {a.iters:,} ({a.iters * GLOBAL_BATCH / TRAIN_BLOCKS:.2f} epochs) · "
           f"train {TRAIN_BLOCKS:,} sequences of {BLOCK:,} tokens · global batch "
           f"{GLOBAL_BATCH:,} sequences · {tokens / 1e9:.2f} G tokens seen · metric val loss")

    # --- the table ---
    rows = []
    for size in SIZE_ORDER:
        for lr in sorted({lr for s, lr in done if s == size}):
            run = done[(size, lr)]
            mean, start, n = window(run, a.last, a.iters, a.every)
            bx, by = best_point(run["steps"], run["val"])
            rows.append((size, lr, mean, by, bx, n, start, window_sd(run, a.last, a.iters, a.every)))
    width = max(len(s) for s, *_ in rows) if rows else 6
    print(f"window: the last {a.last} evaluations, steps {rows[0][6]:,}-{a.iters:,}" if rows else "")
    print(f"{'size':<{width+2}}{'lr':>10}{'window mean':>14}{'sd':>9}{'best':>10}{'at':>8}"
          f"{'best-mean':>11}")
    for size, lr, mean, by, bx, n, start, sd in rows:
        print(f"{size:<{width+2}}{lr:>10g}{mean:>14.4f}{sd:>9.4f}{by:>10.4f}{bx:>8,}"
              f"{by - mean:>11.4f}")

    # Is a gap between two arms bigger than the arms move inside the window? Printed for
    # the pairs the comparison rests on: each size's best rate, in size order.
    bests = []
    for size in SIZE_ORDER:
        same = [r for r in rows if r[0] == size]
        if same:
            bests.append(min(same, key=lambda r: r[2]))
    if len(bests) > 1:
        print("\neach size's best rate, and whether the step to the next size clears the noise:")
        for i, r in enumerate(bests):
            line = f"  {r[0]:<7} lr {r[1]:<8g} window {r[2]:.4f}  sd {r[7]:.4f}"
            if i:
                gap = bests[i - 1][2] - r[2]
                scale = max(r[7], bests[i - 1][7])
                line += (f"  gap from {bests[i - 1][0]} {gap:+.4f} = {abs(gap) / scale:.1f}x "
                         f"the larger sd")
            print(line)

    incomplete = {k: r for k, r in runs.items() if not r["complete"]}
    if incomplete:
        print("\nstill running, left out of the table:")
        for (size, lr), r in sorted(incomplete.items()):
            print(f"  {size} lr {lr:g}: at {max(r['steps']):,} of {a.iters:,}")

    # --- the TSV: every evaluation of every arm, complete or not ---
    tsv = os.path.join(a.out_dir, "molnl-gpt2-grid-val-loss.tsv")
    with open(tsv, "w") as fh:
        fh.write("# molecule_nat_lang GPT-2 learning-rate grid, val loss at every evaluation\n")
        fh.write(f"# {pre.replace(chr(0xb7), chr(0x7c))}\n")
        fh.write(f"# unigram floor for next-token prediction on this corpus: {UNIGRAM_FLOOR}\n")
        fh.write(f"# window_mean is the mean of the last {a.last} evaluations; blank while a run "
                 f"has not reached {a.iters}\n")
        fh.write("size\tlearning_rate\tstep\tval_loss\ttrain_loss\tcomplete\twindow_mean\n")
        for size in SIZE_ORDER:
            for lr in sorted({lr for s, lr in runs if s == size}):
                run = runs[(size, lr)]
                mean, _, _ = window(run, a.last, a.iters, a.every) if run["complete"] else (None, 0, 0)
                for s, v, t in zip(run["steps"], run["val"], run["train"]):
                    fh.write(f"{size}\t{lr:g}\t{s}\t{v:.6f}\t{t:.6f}\t"
                             f"{int(run['complete'])}\t{'' if mean is None else f'{mean:.6f}'}\n")

    # --- one figure per size, rates overlaid ---
    made = []
    for size in SIZE_ORDER:
        arms = [(lr, runs[(size, lr)]) for lr in sorted({lr for s, lr in runs if s == size})]
        if not arms:
            continue
        fig, ax = new_figure()
        items = []
        for i, (lr, run) in enumerate(arms):
            colour = ("#1f6fb4", "#c8571b", "#2e7d32", "#7b1fa2", "#00838f")[i % 5]
            ax.plot(run["steps"], run["val"], color=colour, linewidth=1.3)
            bx, by = best_point(run["steps"], run["val"])
            mean, start, _ = window(run, a.last, a.iters, a.every)
            label = (f"lr {lr:g}: window {mean:.4f}" if run["complete"]
                     else f"lr {lr:g}: running, at {max(run['steps']):,}")
            items.append((bx, by, label, colour))
        ax.set_yscale("log")
        axes(ax, "val loss", xlabel="iteration")
        fit_y(ax, [r["val"] for _, r in arms])
        drew = floor_line(ax, UNIGRAM_FLOOR, "unigram floor")
        label_bests(ax, items, room=0.42)
        note = (f"{pre}\nMarks are each run's best point; the label carries the window mean, which "
                f"is what the arms are judged on (last {a.last} evaluations, steps "
                f"{a.iters - (a.last - 1) * a.every:,}-{a.iters:,})."
                + ("" if drew else f" The unigram floor {UNIGRAM_FLOOR} is off this axis."))
        made.append(save(fig, ax, f"molecule_nat_lang GPT-2 {size} - learning-rate grid", note,
                         os.path.join(a.out_dir, f"gpt2-grid-{size}.png")))

    # --- the summary: window mean against rate, one line per size ---
    if rows:
        fig, ax = new_figure()
        for size in SIZE_ORDER:
            pts = sorted((lr, mean) for s, lr, mean, *_ in rows if s == size)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4,
                    color=SIZE_COLOUR[size], linewidth=1.3, label=size)
            lo = min(pts, key=lambda p: p[1])
            ax.annotate(f"{size} {lo[0]:g}", lo, textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color=SIZE_COLOUR[size])
        ax.set_xscale("log")
        axes(ax, f"mean val loss over the last {a.last} evaluations", xlabel="learning rate")
        note = (f"{pre}\nEach point is one run. A size whose lowest point is at an end of its line "
                "has not been bracketed.")
        made.append(save(fig, ax, "molecule_nat_lang GPT-2 - window mean against learning rate",
                         note, os.path.join(a.out_dir, "gpt2-grid-window-means.png")))

    print()
    for p in made + [tsv]:
        print(f"  {os.path.basename(p)}  {os.path.getsize(p):,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
