#!/usr/bin/env python3
"""Loss against model size with every seed drawn, so a gap is read against the spread.

A single run per size invites a reading the data does not carry. On this grid the step
from medium to large is 0.002 and the three seeds of one size cover 0.009 to 0.032, so
the figure has to show both or it argues for an ordering that is not there.

Each size is drawn as its seeds -- one marker per run -- with a bar spanning them, at the
compute that size used (C = 6ND in petaflop/s-days, N excluding the embedding tables).
The losses come from scoring each checkpoint on the whole validation split, not from the
training-time evaluation, which resamples its batches and carries 0.017 to 0.022 of noise
per point.

    python scripts/molnl_seed_spread.py --scores <json> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import japanize_matplotlib                            # noqa: E402,F401
import matplotlib.pyplot as plt                       # noqa: E402

INK, INK_2, INK_3 = "#0f161a", "#53626c", "#8695a0"
SURFACE, RULE = "#fcfcfb", "#d7e0e5"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
def rate_label(tag):
    mantissa, exponent = tag.split("e")
    return f"{mantissa.replace('p', '.')}e-{exponent}"


PF_DAY = 8.64e19                      # 1e15 FLOP/s x 86,400 s
SIZE_ORDER = ("small", "medium", "large", "xl")
# Non-embedding parameter counts, from the checkpoints (job 147702).
NON_EMBED = {"small": 84_953_856, "medium": 302_040_064,
             "large": 707_882_240, "xl": 1_474_715_200}
KEY = re.compile(r"^(?P<size>small|medium|large|xl)_(?P<iters>\d+)_lr(?P<tag>[0-9p]+e\d)"
                 r"(?:_seed(?P<seed>\d+))?$")


BERT_NON_EMBED = {"small": 85_698_392, "medium": 303_413_336, "large": 575_231_192,
                  "xl": 1_478_176_664}
BERT_NAME = re.compile(r"^bert_(?P<size>small|medium|large|xl)_24k_lr(?P<tag>[0-9p]+e\d)"
                       r"(?:_seed(?P<seed>\d+))?$")


def collect_bert(root, arms, keep_tag=True):
    """{size: [(seed, window mean), ...]} from each run's own trainer_state.json.

    The window is the last ten evaluations, as everywhere else in this grid: with the
    evaluation mask fixed, the spread inside one run is 0.0002 and the mean is a cleaner
    statistic than the single last point.
    """
    import glob as _glob
    out = {}
    for name in arms:
        run_dir = os.path.join(root, name)
        m = BERT_NAME.match(name)
        cks = sorted(_glob.glob(os.path.join(run_dir, "checkpoint-*")),
                     key=lambda p: int(p.rsplit("-", 1)[1]))
        if not m or not cks:
            print(f"  ! skipped {name}")
            continue
        history = json.load(open(os.path.join(cks[-1], "trainer_state.json")))["log_history"]
        points = sorted((e["step"], e["eval_loss_mask"]) for e in history
                        if "eval_loss_mask" in e)
        key = (m["size"], m["tag"]) if keep_tag else m["size"]
        out.setdefault(key, []).append(
            (int(m["seed"] or 42), st.mean([v for _, v in points[-10:]])))
    for size in out:
        out[size].sort()
    return out


def collect(scores, tokens):
    """{size: [(seed, loss), ...]} for the rate each size was repeated at."""
    out = {}
    for name, row in scores["results"].items():
        m = KEY.match(name)
        if not m:
            continue
        out.setdefault(m["size"], []).append((int(m["seed"] or 42), row["loss"]))
    for size in out:
        out[size].sort()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", help="the re-scoring JSON (GPT-2)")
    ap.add_argument("--bert-root", help="draw BERT instead, from the run directories")
    ap.add_argument("--bert-arms", nargs="+", default=[],
                    help="run names to draw, grouped by size from the name")
    ap.add_argument("--representative", nargs="+", default=[], metavar="SIZE=TAG",
                    help="which rate stands for each size on the line, e.g. medium=3e4. "
                         "Other rates of that size are drawn but not joined.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--tokens-per-step", type=int, default=2560 * 1024)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    tokens = a.iters * a.tokens_per_step
    scores = json.loads(open(a.scores, encoding="utf-8").read()) if a.scores else None
    if a.bert_root:
        # Keyed by (size, rate): one size can carry the rate that did best and the rate
        # chosen to stand for it, and they are not the same arm here.
        by_rate = collect_bert(a.bert_root, a.bert_arms)
        rep = dict(x.split("=", 1) for x in a.representative)
        by_size = {size: runs for (size, tag), runs in by_rate.items()
                   if rep.get(size, tag) == tag}
        others = {(size, tag): runs for (size, tag), runs in by_rate.items()
                  if rep.get(size, tag) != tag}
        label = "bert"
    elif scores is not None:
        by_size, label = collect(scores, tokens), "gpt2"
    else:
        ap.error("pass --scores for GPT-2 or --bert-root for BERT")

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    means = []
    if a.bert_root:
        # Rates a size ran that are not its representative: drawn, labelled, not joined.
        for (size, tag), runs in sorted(others.items()):
            x = 6 * BERT_NON_EMBED[size] * tokens / PF_DAY
            values = [v for _, v in runs]
            ax.plot([x, x], [min(values), max(values)], color=INK_3, linewidth=2.2,
                    alpha=.3, zorder=2)
            for _seed, value in runs:
                ax.plot([x], [value], marker="s", markersize=6, color=SURFACE,
                        markeredgecolor=INK_3, markeredgewidth=1.4, zorder=4)
            ax.annotate(f"lr {rate_label(tag)}\n（代表に採らない）",
                        (x, st.mean(values)), textcoords="offset points", xytext=(10, -4),
                        fontsize=7.2, color=INK_3)
    for i, size in enumerate([s for s in SIZE_ORDER if s in by_size]):
        runs = by_size[size]
        x = 6 * (BERT_NON_EMBED if a.bert_root else NON_EMBED)[size] * tokens / PF_DAY
        values = [v for _, v in runs]
        colour = SERIES[i % len(SERIES)]
        ax.plot([x, x], [min(values), max(values)], color=colour, linewidth=2.6,
                alpha=.35, solid_capstyle="round", zorder=2)
        # Two of three seeds can land within 0.001 of each other; one offset for all
        # three writes the labels on top of one another.
        order = sorted(range(len(runs)), key=lambda k: runs[k][1])
        offset = {k: dy for k, dy in zip(order, (-11, -3, 6))}
        for k, (seed, value) in enumerate(runs):
            ax.plot([x], [value], marker="o", markersize=7, color=colour,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
            ax.annotate(f"seed {seed}", (x, value), textcoords="offset points",
                        xytext=(9, offset[k]), fontsize=7.2, color=colour)
        mean = st.mean(values)
        means.append((x, mean, size, max(values) - min(values)))
        ax.plot([x * .93, x * 1.07], [mean, mean], color=colour, linewidth=1.4, zorder=5)
        print(f"  {size:<7} C={x:.4g} PF-days  mean={mean:.4f}  "
              f"spread={max(values) - min(values):.4f}  {[f'{v:.4f}' for v in values]}")
    ax.plot([m[0] for m in means], [m[1] for m in means], color=INK_3, linewidth=1.2,
            linestyle=(0, (5, 3)), zorder=3, label="3 seed の平均")

    for j in range(1, len(means)):
        gap = means[j - 1][1] - means[j][1]
        scale = max(means[j][3], means[j - 1][3])
        ax.annotate(f"{gap:+.4f}\n= 幅の {abs(gap) / scale:.2f} 倍",
                    ((means[j - 1][0] * means[j][0]) ** .5, (means[j - 1][1] + means[j][1]) / 2),
                    textcoords="offset points", xytext=(0, 16), ha="center", fontsize=7.6,
                    color=INK_2)

    ax.set_xscale("log")
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(.8)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3, width=.8)
    ax.grid(color=RULE, linewidth=.6, alpha=.7)
    ax.set_axisbelow(True)
    ax.set_xticks([m[0] for m in means])
    table = BERT_NON_EMBED if a.bert_root else NON_EMBED
    ax.set_xticklabels([f"{m[2]}\n{table[m[2]] / 1e6:.0f}M" for m in means])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("学習に使った計算量（PF-days、C = 6ND）", color=INK_2, fontsize=9)
    ax.set_ylabel("eval_loss_mask（nats/token）" if a.bert_root
                  else "valid 全件での損失（nats/token）", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper right")

    import textwrap
    caption = (
        "前提: max_steps 24,000 / 193.1 epoch / 訓練データ 325.8M トークン（318,118 系列 × 1,024）/ "
        "グローバルバッチ 2,560 系列（2.62M トークン/step）/ 学習トークン 62.9G。"
        "縦軸は最後の 10 評価点の平均（評価のマスク位置は固定、1 run 内の揺れは 0.0002）。"
        "横軸は C = 6ND（N は非埋め込みパラメータ数）を 8.64e19 で割った PF-days。"
        "縦棒は seed の幅、横棒は平均。破線で結んだのは各サイズの代表（small 1e-3・medium 3e-4・"
        "large 1.7e-4）。medium は 4.1e-4 のほうが低い（0.0558、2 本）が、3 seed のうち 1 本"
        "（seed 17）が step 4,000 の 0.1540 のあと発散するため代表に採らず、白い四角で別に描いた。"
        "medium の代表 3e-4 は 1 本だけなので縦棒が無い。何も学ばないモデルの 3.8638 は軸の外。"
        if a.bert_root else
        "前提: max_iters 1,500 / 12.07 epoch / 訓練データ 325.8M トークン（318,118 系列 × 1,024）/ "
        "グローバルバッチ 2,560 系列（2.62M トークン/step）/ 学習トークン 3.93G。"
        "縦軸は各 run の checkpoint を valid 全件（1,631 系列・1,668,513 トークン）で採点し直した次トークン損失。"
        "訓練中の評価は毎回系列を引き直しており 1 点あたり 0.017〜0.022 の雑音があるため、その値ではない。"
        "横軸は C = 6ND（N は非埋め込みパラメータ数）を 8.64e19 で割った PF-days。"
        "縦棒は 3 seed（1・17・42）の幅、横棒は平均。各サイズは自分の最良の学習率"
        "（small 1.2e-3・medium 6e-4・large 3e-4・xl 3e-4）で走らせたもの。"
        "何も学ばないモデルの 4.6514 は軸の外。")
    fig.suptitle("molecule_nat_lang BERT —— medium から large は seed の幅に埋もれる"
                 if a.bert_root else
                 "molecule_nat_lang GPT-2 —— medium 以上ではサイズの差が seed の幅に埋もれる",
                 fontsize=13.5, fontweight="bold", color=INK, x=.012, ha="left", y=.985)
    wrapped = "\n".join(textwrap.wrap(caption, width=118))
    fig.subplots_adjust(bottom=.10 + .035 * (wrapped.count("\n") + 1))
    fig.text(.012, .012, wrapped, fontsize=7.6, color=INK_3, va="bottom", linespacing=1.6)
    fig.patch.set_facecolor(SURFACE)
    path = os.path.join(a.out_dir, f"{label}-seed-spread.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(path)}")

    tsv = os.path.join(a.out_dir, f"molnl-{label}-seed-scores.tsv")
    if a.bert_root:
        with open(tsv, "w") as fh:
            fh.write("# molecule_nat_lang BERT, mean of the last ten evaluations per run\n")
            fh.write("# max_steps 24,000 | global batch 2,560 sequences | 62.9 G tokens seen\n")
            fh.write("size\tseed\tparams_non_embedding\tpf_days\twindow_mean\n")
            for size, runs in by_size.items():
                for seed, value in runs:
                    fh.write(f"{size}\t{seed}\t{BERT_NON_EMBED[size]}\t"
                             f"{6 * BERT_NON_EMBED[size] * tokens / PF_DAY:.4f}\t{value:.6f}\n")
        print(f"  wrote {os.path.basename(tsv)}")
        return 0
    with open(tsv, "w") as fh:
        fh.write("# molecule_nat_lang GPT-2, each checkpoint scored on the whole valid split\n")
        fh.write(f"# {scores['split']} split, {scores['sequences_in_split']:,} sequences | "
                 "max_iters 1,500 | global batch 2,560 sequences | 3.93 G tokens seen\n")
        fh.write("# N is the non-embedding parameter count; compute is C = 6ND in PF-days\n")
        fh.write("size\tseed\tlearning_rate\tparams_non_embedding\tpf_days\tvalid_loss"
                 "\tbest_val_logged\tcheckpoint_iter\n")
        for name, row in sorted(scores["results"].items()):
            m = KEY.match(name)
            if not m:
                continue
            size = m["size"]
            lr = float(m["tag"].replace("p", ".").replace("e", "e-"))
            fh.write(f"{size}\t{int(m['seed'] or 42)}\t{lr:g}\t{NON_EMBED[size]}\t"
                     f"{6 * NON_EMBED[size] * tokens / PF_DAY:.4f}\t{row['loss']:.6f}\t"
                     f"{row['best_val_loss']:.6f}\t{row['checkpoint_iter']}\n")
    print(f"  wrote {os.path.basename(tsv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
