"""Draw the genome loss curves, one figure per series, from the TSV.

Every number here is read from the TSV that `genome_loss_curves.py` wrote, so
the figures and the table cannot disagree. Nothing is thinned: a curve drawn
from decimated points has the wrong shape at the turn, which is the part the
figure exists to show.

**Colour carries the experiment's axis, not the run's identity.** Twenty-one
subsets cannot be told apart by twenty-one hues, and the question is not which
seed did what -- it is whether `global_random` sits apart from
`eukaryote_matched`. So the three corpus families get three colours and the
seeds within a family share one. A named run can still be found: the best point
of each is marked, and the TSV holds every value.

**The axis is set by the curves, never clipped to them.** The order asks that
no line leave the frame, so the limits come from the full extent of the series
being drawn, first evaluation point included -- for GPT-2 that is the value at
iteration 0, near ln(10), which is why the last-20% figure is the one to read
the turn off. A run that collapsed would be drawn separately rather than widen
everyone else's axis; none did, so no figure is split.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401,E402  (no CJK font is installed system-wide)
import matplotlib.pyplot as plt          # noqa: E402

# Three families, three colours. Checked with scripts/validate_palette-style
# separation in mind: these stay distinct under the common CVD simulations and
# in greyscale, because the order also differs (dark -> mid -> light).
FAMILY = [("mammal_centered", "mammal_centered", "#1b3a6b"),
          ("eukaryote_matched", "eukaryote_matched_random_seed*", "#c2622d"),
          ("global_random", "global_random_seed*", "#3f8f6f")]
BASELINE_COLOUR = "#8a8a8a"


def family_of(subset):
    if subset == "mammal_centered":
        return "mammal_centered"
    return "eukaryote_matched" if subset.startswith("eukaryote_matched") else "global_random"


def read_tsv(path):
    curves = defaultdict(lambda: defaultdict(list))
    metric = {}
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            curves[row["series"]][row["subset"]].append((int(row["step"]), float(row["value"])))
            metric[row["series"]] = row["metric"]
    for series in curves:
        for subset in curves[series]:
            curves[series][subset].sort()
    return curves, metric


def caption(series, meta, n_runs, metric):
    """The five things a loss figure has to state beside its curves."""
    got = meta.get(series, {})

    def show(key, fmt="{}"):
        value = got.get(key)
        if value is None:
            return "不明"
        if isinstance(value, list):
            return f"{fmt.format(value[0])}〜{fmt.format(value[1])}"
        return fmt.format(value)

    line = (f"{series}  |  {n_runs} 本  |  窓 {show('window', '{:,}')}  |  "
            f"エポック {show('epochs')}  |  step {show('max_steps', '{:,}')}  |  "
            f"グローバルバッチ {show('global_batch', '{:,}')}  |  "
            f"学習率 {show('lr')}  |  評価間隔 {show('eval_every', '{:,}')}  |  {metric}")
    if n_runs > 1:
        # Each run is scored on its own subset's valid split, so the curves are
        # not one task measured 21 ways. Height across subsets is not a ranking;
        # the distance to that subset's own baseline is the comparable quantity.
        line += ("\n各 run は自分の subset の valid で採点している。"
                 "subset をまたいで高さを比べることはできない。"
                 "比べられるのは各 subset 自身の基準線からの差である。")
    return line


def draw(series, runs, metric, meta, baselines, out_path, tail_fraction=None,
         baseline_method=""):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    last_step = max(s for pts in runs.values() for s, _ in pts)
    lo_step = last_step * (1 - tail_fraction) if tail_fraction else 0

    seen, ys, labelled = set(), [], False
    for subset, points in sorted(runs.items()):
        kept = [(s, v) for s, v in points if s >= lo_step]
        if not kept:
            continue
        fam = family_of(subset)
        colour = dict((f[0], f[2]) for f in FAMILY)[fam]
        ax.plot(*zip(*kept), color=colour, linewidth=1.0, alpha=0.75,
                label=fam if fam not in seen else None)
        seen.add(fam)
        ys += [v for _, v in kept]
        best_step, best_value = min(kept, key=lambda p: p[1])
        ax.plot([best_step], [best_value], marker="o", markersize=4.5,
                color=colour, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        if len(runs) <= 3 or tail_fraction:
            labelled = True
            ax.annotate(f"{best_step:,}", (best_step, best_value),
                        textcoords="offset points", xytext=(0, -12),
                        ha="center", fontsize=6.5, color=colour)

    drawn = [baselines[s] for s in runs if s in baselines]
    if drawn:
        # Per-subset values, never one line standing for twenty-one: the band is
        # their full spread, and each subset's own number is in the TSV.
        if len(drawn) == 1:
            ax.axhline(drawn[0], color=BASELINE_COLOUR, linestyle="--", linewidth=1.0,
                       label=f"基準線 {drawn[0]:.4f}（{baseline_method}）")
        else:
            ax.axhspan(min(drawn), max(drawn), color=BASELINE_COLOUR, alpha=0.18,
                       label=f"基準線 subset ごと {min(drawn):.4f}〜{max(drawn):.4f}"
                             f"（{baseline_method}）")
        ys += [min(drawn), max(drawn)]

    span = max(ys) - min(ys)
    # The step labels sit 12 points below their marker, so a figure that draws
    # them needs room underneath or the lowest run's label is cut by the axis.
    ax.set_ylim(min(ys) - span * (0.11 if labelled else 0.04), max(ys) + span * 0.04)
    ax.set_xlim(lo_step, last_step)
    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.xaxis.set_major_formatter(lambda v, _pos: f"{int(v):,}")
    ax.set_title(("最後の 20 %: " if tail_fraction else "") + series, fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    fig.text(0.01, 0.015, caption(series, meta, len(runs), metric), fontsize=6.6,
             color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def draw_epoch_pair(curves, metric, meta, baselines, out_path):
    """3 epochs against 9, on the one subset that has both.

    sat9 is mammal_centered alone, so the run it is placed beside is
    mammal_centered from the 512-window series -- the same subset, the same
    window, the same batch. Any other subset would change two things at once.
    """
    fig, ax = plt.subplots(figsize=(11, 5.2))
    pairs = (("bert-base", "mammal_centered", "3 エポック (bert-base)", "#1b3a6b"),
             ("bert-sat9", "mammal_centered", "9 エポック (bert-sat9)", "#c2622d"))
    ys = []
    for series, subset, label, colour in pairs:
        points = curves.get(series, {}).get(subset)
        if not points:
            print(f"  {series}/{subset} が無いので省いた")
            continue
        ax.plot(*zip(*points), color=colour, linewidth=1.2, label=label)
        ys += [v for _, v in points]
        best_step, best_value = min(points, key=lambda p: p[1])
        ax.plot([best_step], [best_value], marker="o", markersize=5, color=colour,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.annotate(f"{best_step:,} / {best_value:.4f}", (best_step, best_value),
                    textcoords="offset points", xytext=(0, -13), ha="center",
                    fontsize=7, color=colour)
    if "mammal_centered" in baselines:
        value = baselines["mammal_centered"]
        ax.axhline(value, color=BASELINE_COLOUR, linestyle="--", linewidth=1.0,
                   label=f"基準線 {value:.4f}")
        ys.append(value)
    span = max(ys) - min(ys)
    ax.set_ylim(min(ys) - span * 0.11, max(ys) + span * 0.04)
    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.xaxis.set_major_formatter(lambda v, _pos: f"{int(v):,}")
    ax.set_title("mammal_centered: 3 エポックと 9 エポック", fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.text(0.01, 0.015, "窓 512・グローバルバッチ 2,560・学習率 1e-4・"
             "指標 eval_loss_mask。両者とも mammal_centered の 1 本である。",
             fontsize=6.6, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def load_baselines(spec, tally_spec):
    """series=source pairs; a series with no measurement on its own build has none.

    Two shapes, because the two were measured for different reasons. The final
    scorers write one file per subset and take the baseline at exactly the
    positions that paid loss, in the same pass -- that is the campaign's record
    for the 1,026 and 1,024 builds. The composition tally writes one file for
    all subsets and counts the whole split; it is what the 512 build has. The
    two agree to about 0.002 where both exist, but they are different
    measurements and the figure names which one it drew.
    """
    out, method = defaultdict(dict), {}
    for item in spec:
        series, _, directory = item.partition("=")
        for path in sorted(__import__("glob").glob(os.path.join(directory, "*.json"))):
            value = json.load(open(path)).get("degenerate_baseline")
            if isinstance(value, dict):
                value = value.get("mean")
            if value is not None:
                out[series][os.path.splitext(os.path.basename(path))[0]] = float(value)
                method[series] = "採点と同一パス"
    for item in tally_spec:
        series, _, rest = item.partition("=")
        path, _, model = rest.partition(":")
        if not os.path.exists(path):
            continue
        for row in json.load(open(path)):
            got = row.get(model or "bert", {}).get("baseline")
            if got is not None:
                out[series][row["subset"]] = float(got)
                method[series] = "組成の集計"
    return out, method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--meta", default="")
    ap.add_argument("--baselines", nargs="*", default=[],
                    help="series=directory of <subset>.json holding degenerate_baseline")
    ap.add_argument("--baselines-tally", nargs="*", default=[],
                    help="series=path/to/degenerate-baselines.json:model")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="genome-loss")
    args = ap.parse_args()

    # japanize_matplotlib set the family at import; only the minus sign is left,
    # which IPAexGothic renders as a full-width dash without this.
    plt.rcParams["axes.unicode_minus"] = False

    curves, metric = read_tsv(args.tsv)
    meta = json.load(open(args.meta)) if args.meta and os.path.exists(args.meta) else {}
    baselines, method = load_baselines(args.baselines, args.baselines_tally)
    os.makedirs(args.out_dir, exist_ok=True)

    for series in sorted(curves):
        base = baselines.get(series, {})
        if not base:
            print(f"  {series}: このビルドで測った基準線が無いので引かない")
        draw(series, curves[series], metric[series], meta, base,
             os.path.join(args.out_dir, f"{args.prefix}-{series}.png"),
             baseline_method=method.get(series, ""))
        draw(series, curves[series], metric[series], meta, base,
             os.path.join(args.out_dir, f"{args.prefix}-{series}-tail20.png"),
             tail_fraction=0.20, baseline_method=method.get(series, ""))

    draw_epoch_pair(curves, "eval_loss_mask", meta, baselines.get("bert-base", {}),
                    os.path.join(args.out_dir, f"{args.prefix}-epoch-3-vs-9.png"))


if __name__ == "__main__":
    main()
