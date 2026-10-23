#!/usr/bin/env python3
"""Where the validation loss bottoms out, against how long the schedule runs.

The GPT-2 grid ran 1,500 iterations, which is 12.07 epochs of this corpus against the
193.1 the BERT grid runs, and the question was whether the budget was simply too small.
Doubling it answers no: every size reaches its best value between steps 1,400 and 1,600
and is worse afterwards.

This is not the size-comparison cut approved on 2026-09-28 (a panel per learning rate
with the sizes overlaid), because what is being compared here is the schedule and not the
size: one panel per size, the two schedules drawn in it, and the minimum of each marked.
The sizes are not to be read against each other off this figure -- the spread across
seeds of one arm, 0.0153 to 0.0322, is wider than the gaps between the sizes.

    python scripts/molnl_schedule_figure.py --runs-root <gpt2-output-grid> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401,E402  (registers a CJK font)
import matplotlib.pyplot as plt  # noqa: E402

from molnl_loss_figures import INK, axes  # noqa: E402

PF_DAY = 8.64e19
TOKENS_PER_STEP = 2_621_440
FLOOR = 4.6514  # next-token unigram, train frequencies scored on valid
SIZES = ("small", "medium", "large", "xl")
RATES = {"small": "1.2e-3", "medium": "6e-4", "large": "3e-4", "xl": "3e-4"}
SCHEDULES = (("1500_fixedval", "1,500 iteration", "#1f6fb4", "-"),
             ("3000", "3,000 iteration", "#c8571b", "-"))


def series(run_dir):
    out = {}
    for path in glob.glob(os.path.join(run_dir, "logging_*.csv")):
        for row in csv.reader(open(path)):
            if row and row[0].strip().isdigit():
                out[int(row[0])] = float(row[2])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--sizes", required=True, help="the measured parameter counts")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--caption", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    counts = json.load(open(a.sizes))
    n_of = {k.split("_")[1]: v["non_embedding"] for k, v in counts.items()
            if v["arch"] == "gpt2"}

    fig, panels = plt.subplots(1, 4, figsize=(15.2, 4.6), sharey=True)
    rows = []
    for ax, size in zip(panels, SIZES):
        n = n_of[size]
        for suffix, label, colour, style in SCHEDULES:
            points = series(os.path.join(a.runs_root, f"gpt2_{size}_{suffix}"))
            if not points:
                continue
            steps = sorted(points)
            xs = [6 * n * s * TOKENS_PER_STEP / PF_DAY for s in steps]
            ax.plot(xs, [points[s] for s in steps], style, color=colour, linewidth=1.6,
                    label=label)
            best = min(steps, key=lambda s: points[s])
            ax.plot([6 * n * best * TOKENS_PER_STEP / PF_DAY], [points[best]],
                    marker="o", ms=6, color=colour, markeredgecolor="white",
                    markeredgewidth=1.1, zorder=5)
            rows.append((size, label, best, points[best], steps[-1], points[steps[-1]]))
            if suffix == "3000":
                # In the empty upper corner rather than beside the mark: the minimum
                # sits low and near the right edge, where a label either left the axes
                # or lay across the tick labels. One marked point per panel, so no
                # leader line is needed to say which point it describes.
                ax.annotate(f"3,000 の最小 {points[best]:.4f}（step {best:,}）\n"
                            f"最終 {points[steps[-1]]:.4f}",
                            (0.97, 0.96), xycoords="axes fraction", ha="right",
                            va="top", fontsize=8, color=colour)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{size}  {n / 1e6:.1f}M  lr {RATES[size]}", fontsize=10.5,
                     color=INK, loc="left", pad=6)
        axes(ax, "val loss（nats/token）" if size == "small" else "",
             "学習に使った計算量（PF-days、C = 6ND）")
    panels[0].legend(frameon=False, fontsize=9, loc="lower left")
    for ax in panels:
        ax.axhline(FLOOR, color="#888888", linestyle=":", linewidth=1)
    panels[-1].annotate(f"何も学ばない場合 {FLOOR}", (0.985, FLOOR),
                        xycoords=("axes fraction", "data"), ha="right", va="bottom",
                        fontsize=7.5, color="#666666")

    import textwrap
    title = "1,500 iteration は既に最適点を過ぎている —— 4 サイズとも step 1,400〜1,600 で底を打つ"
    fig.suptitle(title, fontsize=11.5, color=INK, x=0.008, ha="left")
    caption = open(a.caption, encoding="utf-8").read().strip()
    line_h = (7.5 * 1.35) / (fig.get_size_inches()[1] * 72)
    lines = [ln for raw in caption.split("\n")
             for ln in textwrap.wrap(raw, int(fig.get_size_inches()[0] * 10))]
    fig.text(0.008, 0.012, "\n".join(lines), fontsize=7.5, color="#666666", va="bottom")
    fig.tight_layout(rect=(0.008, 0.012 + line_h * len(lines) + 0.03, 0.995, 0.93))
    path = os.path.join(a.out_dir, "molnl-gpt2-schedule.png")
    fig.savefig(path, dpi=150)
    tsv = os.path.join(a.out_dir, "molnl-gpt2-schedule.tsv")
    with open(tsv, "w") as fh:
        fh.write("size\tschedule\tbest_step\tbest_val_loss\tlast_step\tlast_val_loss\n")
        for row in rows:
            fh.write("\t".join(str(v) for v in row) + "\n")
    print(f"  wrote {path}\n  wrote {tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
