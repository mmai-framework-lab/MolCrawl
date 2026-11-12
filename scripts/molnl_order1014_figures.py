#!/usr/bin/env python3
"""The two figures the 2026-10-14 order's answers are read off.

What each one is for:

  attained   where each size ended up, against the compute it took to get there, with the
             original run and the re-run drawn as two marks and the seed spread of the
             same arm as the bar. The question §2 asks is whether holding the validation
             set changes the per-size picture, which is a question about distances, so
             the distance the seeds already cover has to be on the same figure.
  gap        the training loss beside the validation loss, which is what §4's table
             carries and what says whether these runs are fitting their corpus.

The attained and gap figures keep the two architectures in separate panels with their own
y axes: eval_loss_mask scores the 20% of positions BERT masked and val_loss scores every
token GPT-2 predicted, and no axis makes those one number.

There was a third figure here, Recall@1 and MRR for the seven models on one axis. The
2026-11-10 judgment removes it twice over: 3.3 deletes the measurements it was drawn from,
and 3.1 refuses the comparison it put on one axis, because the two architectures' scores
come from different estimators even where the candidate set is the same.

    python scripts/molnl_order1014_figures.py --table <tsv> --sizes <json> \
        --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401,E402  (registers a CJK font)
import matplotlib.pyplot as plt  # noqa: E402

from molnl_loss_figures import INK, SIZE_COLOUR, axes  # noqa: E402

PF_DAY = 8.64e19
WINDOW = 10
# molnl valid split, [MASK] rate 0.2, on training_ready_hf_dataset_shuffled:
# report/mlm-baselines-20261102/full/molnl.json (unigram, fit on 200,000 train
# rows and scored on the 1,631 valid rows). The figures' y axis is the valid
# eval_loss_mask, so the floor has to be the valid one too. The 3.8638 that stood
# here was measured on the train split (job 22503, 2026-08-12) and sits 0.78 below.
FLOOR = {"bert": ("何も学ばない場合（[MASK] 位置のユニグラム）", 4.6449),
         "gpt2": ("何も学ばない場合（次トークンのユニグラム）", 4.6514)}
# (arch, size, the run as it was, the run repeated under the 2026-10-14 order)
PAIRS = [
    ("bert", "small", "bert_small_24k_lr1e3", "bert_small_24k_rerun", "1e-3"),
    ("bert", "medium", "bert_medium_24k_lr3e4", "bert_medium_24k_rerun", "3e-4"),
    ("bert", "large", "bert_large_24k_lr1p7e4", "bert_large_24k_rerun", "1.7e-4"),
    ("gpt2", "small", "gpt2_small_1500_lr1p2e3", "gpt2_small_1500_fixedval", "1.2e-3"),
    ("gpt2", "medium", "gpt2_medium_1500_lr6e4", "gpt2_medium_1500_fixedval", "6e-4"),
    ("gpt2", "large", "gpt2_large_1500_lr3e4", "gpt2_large_1500_fixedval", "3e-4"),
    ("gpt2", "xl", "gpt2_xl_1500_lr3e4", "gpt2_xl_1500_fixedval", "3e-4"),
]
ARCH_NAME = {"bert": "BERT（eval_loss_mask）", "gpt2": "GPT-2（val_loss）"}
DIRECTION_NAME = {"molecule_to_text": "分子 → 説明文", "text_to_molecule": "説明文 → 分子"}


def read_table(path):
    val, train = defaultdict(dict), defaultdict(dict)
    for row in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
        if row["metric"] == "val_loss_full_split":
            continue
        target = val if row["split"] == "validation" else train
        target[row["run"]][int(row["step"])] = float(row["value"])
    return val, train


def window_mean(points):
    steps = sorted(points)[-WINDOW:]
    return sum(points[s] for s in steps) / len(steps)


def pfdays(step, non_embedding):
    return 6 * non_embedding * step * 2_621_440 / PF_DAY


def seed_spread(val, base):
    """max - min over the arm's seeds, or None when the arm has fewer than two that held.

    A seed that diverged is not spread, it is a different outcome, so an arm whose other
    seeds left the floor behind is reported as having no measured spread rather than a
    huge one.
    """
    kept = []
    for name in (base, base + "_seed1", base + "_seed17"):
        if name not in val:
            continue
        mean = window_mean(val[name])
        if mean < 1.0:        # the diverged seeds sit three orders above this
            kept.append(mean)
    return (max(kept) - min(kept)) if len(kept) >= 2 else None


def fig_attained(val, sizes, out_dir, caption):
    fig, panels = plt.subplots(1, 2, figsize=(11.2, 5.6))
    for ax, arch in zip(panels, ("bert", "gpt2")):
        drawn = []
        for a, size, original, rerun, rate in PAIRS:
            if a != arch or original not in val or rerun not in val:
                continue
            n = sizes[rerun]["non_embedding"]
            x = pfdays(max(val[rerun]), n)
            y_old, y_new = window_mean(val[original]), window_mean(val[rerun])
            spread = seed_spread(val, original)
            colour = SIZE_COLOUR[size]
            if spread:
                ax.errorbar([x], [y_old], yerr=[[spread / 2], [spread / 2]], fmt="none",
                            ecolor=colour, elinewidth=1.2, capsize=4, alpha=.65, zorder=2)
            ax.plot([x], [y_old], marker="o", ms=7, color=colour, zorder=4,
                    markeredgecolor="white", markeredgewidth=1.1)
            ax.plot([x], [y_new], marker="D", ms=6, markerfacecolor="white",
                    markeredgecolor=colour, markeredgewidth=1.6, zorder=5)
            ax.annotate(f"{size} {n / 1e6:.1f}M\nlr {rate}", (x, max(y_old, y_new)),
                        textcoords="offset points", xytext=(9, 4), ha="left",
                        fontsize=8, color=colour)
            drawn += [y_old, y_new]
        ax.set_xscale("log")
        ax.set_yscale("log")
        lo, hi = min(drawn), max(drawn)
        span = math.log10(hi) - math.log10(lo)
        ax.set_ylim(10 ** (math.log10(lo) - span * .55), 10 ** (math.log10(hi) + span * .35))
        axes(ax, "到達値（nats/token）" if arch == "bert" else "",
             "学習に使った計算量（PF-days、C = 6ND）")
        ax.set_xmargin(0.22)
        ax.autoscale_view()
        ax.set_title(ARCH_NAME[arch], fontsize=10.5, color=INK, loc="left", pad=6)
        name, value = FLOOR[arch]
        # Below the data: every point sits in the upper half, and this line is far above
        # the axis range, so it is written rather than drawn.
        ax.annotate(f"{name} {value}（軸の外）", (0.02, 0.04), xycoords="axes fraction",
                    fontsize=7.5, color="#666666", va="bottom")
    marks = [plt.Line2D([], [], marker="o", color="#777777", linestyle="none"),
             plt.Line2D([], [], marker="D", markerfacecolor="white", linestyle="none",
                        markeredgecolor="#777777", markeredgewidth=1.6),
             plt.Line2D([], [], marker="|", color="#777777", linestyle="none", ms=12)]
    return save_two(fig, "検証を固定して走り直しても、どのサイズも seed のばらつきの内側にとどまる",
                    caption, os.path.join(out_dir, "molnl-attained-vs-compute.png"),
                    legend=(marks, ["元の run", "再走行（検証を固定）", "同じアームの seed 幅"]))


def save_two(fig, title, caption, path, legend=None):
    """save() for a figure with more than one panel: the title goes above them all.

    The caption is wrapped to the figure's own width rather than to a fixed column, which
    once ran the last line off the right edge, and the panels are given a left margin so
    the y label is not clipped by it.
    """
    import textwrap
    fig.suptitle(title, fontsize=11.5, color=INK, x=0.012, ha="left")
    # Japanese glyphs are full width, so a line holds about ten of them per inch at
    # 7.5pt -- counting them as Latin characters ran the caption off the right edge.
    width = int(fig.get_size_inches()[0] * 10)
    lines = [ln for raw in caption.split("\n") for ln in textwrap.wrap(raw, width)]
    # The caption is drawn upward from the bottom edge; the legend goes above its top,
    # and the panels above that. A guessed line height put the legend through the caption
    # twice, so the height is taken from the font and the figure's own size.
    line_h = (7.5 * 1.35) / (fig.get_size_inches()[1] * 72)
    caption_top = 0.012 + line_h * len(lines)
    bottom = caption_top
    if legend is not None:
        handles, labels = legend
        fig.legend(handles, labels, frameon=False, fontsize=8.5, labelcolor=INK,
                   loc="lower center", bbox_to_anchor=(0.5, caption_top + 0.012),
                   ncol=min(len(labels), 6))
        bottom = caption_top + 0.075
    fig.text(0.012, 0.012, "\n".join(lines), fontsize=7.5, color="#666666", va="bottom")
    fig.tight_layout(rect=(0.012, bottom, 0.995, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_gap(val, train, sizes, out_dir, caption):
    fig, panels = plt.subplots(1, 2, figsize=(11.2, 5.6))
    for ax, arch in zip(panels, ("bert", "gpt2")):
        drawn = []
        for a, size, _, rerun, _ in PAIRS:
            if a != arch or rerun not in val:
                continue
            n = sizes[rerun]["non_embedding"]
            for points, style in ((val[rerun], "-"), (train[rerun], "--")):
                steps = sorted(points)
                ax.plot([pfdays(s, n) for s in steps], [points[s] for s in steps],
                        style, color=SIZE_COLOUR[size], linewidth=1.5)
                drawn += [points[s] for s in steps]
        ax.set_xscale("log")
        ax.set_yscale("log")
        axes(ax, "損失（nats/token）" if arch == "bert" else "",
             "学習に使った計算量（PF-days、C = 6ND）")
        ax.set_title(ARCH_NAME[arch], fontsize=10.5, color=INK, loc="left", pad=6)
        name, value = FLOOR[arch]
        if min(drawn) <= value <= max(drawn):
            ax.axhline(value, color="#888888", linestyle=":", linewidth=1)
            ax.annotate(f"{name} {value}", (0.985, value),
                        xycoords=("axes fraction", "data"), ha="right", va="bottom",
                        fontsize=7.5, color="#666666")
    sizes_drawn = [s for _, s, _, r, _ in PAIRS if r in val]
    handles = [plt.Line2D([], [], color=SIZE_COLOUR[s], linewidth=1.8)
               for s in dict.fromkeys(sizes_drawn)]
    handles += [plt.Line2D([], [], color="#777777", linewidth=1.5),
                plt.Line2D([], [], color="#777777", linewidth=1.5, linestyle="--")]
    labels = list(dict.fromkeys(sizes_drawn)) + ["検証分割", "訓練バッチ"]
    return save_two(fig, "再走行の訓練損失と検証損失",
                    caption, os.path.join(out_dir, "molnl-train-vs-validation.png"),
                    legend=(handles, labels))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", required=True)
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--captions", required=True, help="JSON: {figure: caption}")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    val, train = read_table(a.table)
    sizes = json.load(open(a.sizes))
    caps = json.load(open(a.captions, encoding="utf-8"))
    for path in (fig_attained(val, sizes, a.out_dir, caps["attained"]),
                 fig_gap(val, train, sizes, a.out_dir, caps["gap"])):
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
