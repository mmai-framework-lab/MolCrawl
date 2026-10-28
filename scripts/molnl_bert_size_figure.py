#!/usr/bin/env python3
"""Where each BERT size ends up, with xl's rates beside the three that learn.

xl has been the open question since 2026-10-01: it holds at low rates and diverges above
8.5e-5, and the four rates that hold have now run the full 24,000 steps. This puts them on
one axis with the three smaller sizes so the size of the gap is readable -- it is a factor
of 50, not a few percent, and the four xl rates line up in order rather than scattering.

Two definitions of eval_loss_mask meet on this figure. PR #307 made it the sum over ranks
where it had been rank 0's quarter shard; the xl runs crossed that change mid-run and the
smaller sizes predate it. The shift measures 0.037 to 0.041 at xl's level, and at the
level of the smaller sizes it is inside their seed spread -- medium's seed 42 reads 0.0570
under the old definition and 0.0574 under the new one, against a spread of 0.0015 over its
three seeds. The caption says so; on a log axis spanning 0.05 to 3.9 neither is visible.

    python scripts/molnl_bert_size_figure.py --table <tsv> --sizes <json> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import textwrap
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401,E402
import matplotlib.pyplot as plt  # noqa: E402

from molnl_loss_figures import INK, SIZE_COLOUR, axes  # noqa: E402

FLOOR = 3.8638
WINDOW = 10
# The rate that stands for each size, and for xl every rate that held.
# One definition of eval_loss_mask across the three: the grid's own seed-42 runs all
# predate PR #307. The medium re-run s42r reads 0.0574 under the new definition against
# this run's 0.0570, and the caption says so.
LEARNS = [("small", "bert_small_24k_lr1e3", "1e-3"),
          ("medium", "bert_medium_24k_lr3e4", "3e-4"),
          ("large", "bert_large_24k_lr1p7e4", "1.7e-4")]
XL = [("bert_xl_24k_lr6e5", "6e-5"), ("bert_xl_24k_lr4p3e5", "4.3e-5"),
      ("bert_xl_24k_lr3p1e5", "3.1e-5"), ("bert_xl_24k_lr2p2e5", "2.2e-5")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", required=True)
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--caption", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    series = defaultdict(dict)
    for row in csv.DictReader(open(a.table, encoding="utf-8"), delimiter="\t"):
        if row["split"] == "validation" and row["metric"] == "eval_loss_mask":
            series[row["run"]][int(row["step"])] = float(row["value"])

    def window(run):
        points = series[run]
        steps = sorted(points)[-WINDOW:]
        return sum(points[s] for s in steps) / len(steps)

    counts = json.load(open(a.sizes))
    n_of = {k.split("_")[1]: v["non_embedding"] for k, v in counts.items()
            if v["arch"] == "bert"}

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    rows = []
    for size, run, rate in LEARNS:
        y, n = window(run), n_of[size]
        ax.plot([n], [y], marker="o", ms=9, color=SIZE_COLOUR[size],
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        # Above the point: below it is the x axis and its labels.
        ax.annotate(f"{size}  lr {rate}\n{y:.4f}", (n, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8.5, color=SIZE_COLOUR[size])
        rows.append((size, rate, n, y))
    n_xl = n_of["xl"]
    # The four xl rates land within 0.22 of each other, which is nothing on a log axis
    # spanning two decades, so their labels are stacked rather than placed at the points.
    for i, (run, rate) in enumerate(XL):
        y = window(run)
        ax.plot([n_xl], [y], marker="D", ms=7, markerfacecolor="white",
                markeredgecolor=SIZE_COLOUR["xl"], markeredgewidth=1.6, zorder=5)
        ax.annotate(f"xl  lr {rate}  {y:.4f}", (n_xl, y), textcoords="offset points",
                    xytext=(14, 26 - 17 * i), ha="left", fontsize=8.5,
                    color=SIZE_COLOUR["xl"],
                    arrowprops=dict(arrowstyle="-", color=SIZE_COLOUR["xl"],
                                    lw=0.6, alpha=0.6))
        rows.append(("xl", rate, n_xl, y))
    # To the right of large and well under xl's labels: the only clear strip at this
    # height. The three sizes are one band and the figure has to say so rather than
    # leave three labelled points to be read as an ordering.
    ax.annotate("3 サイズとも区別が付かない\n（差 0.0005 と 0.0026、seed 幅 0.0013〜0.0015 の 0.3〜2.0 倍）",
                (n_of["large"] * 1.9, 0.075), fontsize=8.5, color="#666666",
                ha="left", va="bottom")
    ax.axhline(FLOOR, color="#888888", linestyle="--", linewidth=1.2)
    ax.annotate(f"何も学ばない場合 {FLOOR}", (0.02, FLOOR), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=8.5, color="#666666")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(n_of["small"] * 0.5, n_xl * 7)
    ax.set_ylim(0.04, 7)
    axes(ax, "到達値（nats/token）", "非埋め込みパラメータ数")
    ax.plot([], [], marker="o", color="#777777", linestyle="none", label="学習に入ったサイズ（各自の最良学習率）")
    ax.plot([], [], marker="D", markerfacecolor="white", markeredgecolor="#777777",
            color="#777777", linestyle="none", markeredgewidth=1.6,
            label="xl（発散しなかった 4 つの学習率・いずれも未到達）")
    # Under the floor annotation and above the three points: the corners are taken.
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK,
              bbox_to_anchor=(0.02, 0.78))

    title = "xl は 24,000 step を使い切っても到達せず、小さい 3 サイズは互いに区別が付かない"
    fig.suptitle(title, fontsize=11.5, color=INK, x=0.008, ha="left")
    caption = open(a.caption, encoding="utf-8").read().strip()
    line_h = (7.5 * 1.35) / (fig.get_size_inches()[1] * 72)
    lines = [ln for raw in caption.split("\n")
             for ln in textwrap.wrap(raw, int(fig.get_size_inches()[0] * 10))]
    fig.text(0.008, 0.012, "\n".join(lines), fontsize=7.5, color="#666666", va="bottom")
    fig.tight_layout(rect=(0.008, 0.012 + line_h * len(lines) + 0.03, 0.995, 0.93))
    fig.savefig(os.path.join(a.out_dir, "molnl-bert-size-attained.png"), dpi=150)
    with open(os.path.join(a.out_dir, "molnl-bert-size-attained.tsv"), "w") as fh:
        fh.write("size\tlearning_rate\tnon_embedding_params\twindow_mean\n")
        for row in rows:
            fh.write("\t".join(str(v) for v in row) + "\n")
    print(f"  wrote molnl-bert-size-attained.png and .tsv ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
