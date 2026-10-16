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
import re
from collections import defaultdict


def _pyplot():
    """matplotlib, configured, imported on first use.

    Deliberately not imported at module scope. Which colour a run gets and
    whether a collapsed run is split out are plain Python rules with no
    plotting in them, and the unit suite should be able to reach them on a
    machine with no plotting stack installed -- CI has none, and importing
    matplotlib at the top made those seven tests fail on the import line
    rather than on anything they were checking.

    The backend has to be selected before pyplot is imported, and
    japanize_matplotlib sets the font family as a side effect of its own
    import, so the order here is load-bearing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import japanize_matplotlib  # noqa: F401  (no CJK font is installed system-wide)
    import matplotlib.pyplot as plt
    # japanize_matplotlib set the family; only the minus sign is left, which
    # IPAexGothic renders as a full-width dash without this.
    plt.rcParams["axes.unicode_minus"] = False
    return plt


# Three families, three colours. Checked with scripts/validate_palette-style
# separation in mind: these stay distinct under the common CVD simulations and
# in greyscale, because the order also differs (dark -> mid -> light).
FAMILY = [("mammal_centered", "mammal_centered", "#1b3a6b"),
          ("eukaryote_matched", "eukaryote_matched_random_seed*", "#c2622d"),
          ("global_random", "global_random_seed*", "#3f8f6f")]
BASELINE_COLOUR = "#8a8a8a"

# Used when every run in a series shares one subset -- a learning-rate sweep,
# where the arm is what varies and the family colour would paint all of them
# the same. Ordered dark to light so they also separate in greyscale.
ARM_COLOURS = ["#1b3a6b", "#c2622d", "#3f8f6f", "#7b4b8a", "#8a6d1f",
               "#2f7f8f", "#a03a4a", "#4a4a4a"]


def base_subset(label):
    """The subset a label names, with any learning-rate arm removed.

    A sweep's curves are labelled `mammal_centered-lr3e4` so three of them do
    not share one name, but everything that asks "which subset is this" -- the
    colour family, the baseline lookup, whether heights are comparable -- has
    to see `mammal_centered`. Without this the sweep figures came out labelled
    global_random, drew no baseline, and carried a footnote saying heights
    could not be compared when every run in them is the same subset.
    """
    return re.sub(r"-(stab2?-)?lr[\dp]+e-?\d+(-\w+)?$", "", label)


def colour_map(runs):
    """Colour by corpus family, or by run when the family cannot separate them.

    The production series vary the subset, so the three corpus families carry
    the experiment's axis. A learning-rate sweep varies the arm on one subset,
    where every run is the same family and one colour would cover the lot.
    """
    subsets = {base_subset(s) for s in runs}
    if len(runs) > 1 and len(subsets) == 1 and len(runs) <= len(ARM_COLOURS):
        return {s: ARM_COLOURS[i] for i, s in enumerate(sorted(runs))}, True
    by_family = dict((f[0], f[2]) for f in FAMILY)
    return {s: by_family[family_of(s)] for s in runs}, False


def split_collapsed(runs, baselines):
    """Runs that ended above their own baseline, separated from the rest.

    The order asks that a run which collapsed not share a figure with runs that
    learned: its axis is set by a value the others never reach, and every curve
    is then squeezed into a band where its slope cannot be read. With no
    baseline measured for the series nothing is split -- guessing a threshold
    would be worse than leaving the figure whole.
    """
    if not baselines:
        return runs, {}
    learned, collapsed = {}, {}
    for subset, points in runs.items():
        line = baselines.get(subset) or baselines.get(base_subset(subset)) \
            or max(baselines.values())
        (collapsed if points[-1][1] >= line else learned)[subset] = points
    return (learned, collapsed) if learned and collapsed else (runs, {})


def family_of(subset):
    subset = base_subset(subset)
    if subset.startswith("mammal_centered"):
        return "mammal_centered"
    if subset.startswith("eukaryote_matched"):
        return "eukaryote_matched"
    return "global_random"


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


def tokens_per_step(series, meta):
    """Sequences per step times the length of one, from what the runs recorded.

    A step is not the same amount of data in two series: the 512-window runs
    process half of what the 1,026-window runs do at the same global batch.
    Plotting against step would make the shorter-window series look faster for
    a reason that has nothing to do with learning.
    """
    got = meta.get(series, {})
    window, batch = got.get("window"), got.get("global_batch")
    if isinstance(window, list) or isinstance(batch, list) or not window or not batch:
        return None
    return int(window) * int(batch)


def token_tick(value, _pos=None):
    """A tick on the processed-token axis.

    One decimal below 10 G and none above: a bare "{:,.0f} G" printed two
    different decades as "0 G" apiece. Below 1 G it switches to M, because the
    first evaluation of a short run lands there.
    """
    if value >= 1e10:
        return f"{value / 1e9:,.0f} G"
    if value >= 1e9:
        return f"{value / 1e9:.1f} G"
    return f"{value / 1e6:,.0f} M"


def step_tick(value, _pos=None):
    """A tick on the step axis, for the figures not drawn against tokens."""
    return f"{int(value):,}"


def matplotlib_null_formatter():
    """A formatter that prints nothing, imported late like pyplot itself."""
    import matplotlib.ticker

    return matplotlib.ticker.NullFormatter()


def caption(series, meta, n_runs, metric, many_subsets, per_step, reached=None):
    """The five things a loss figure has to state beside its curves.

    `reached` is the furthest step any curve in the figure actually reached.
    It is not the schedule's end: a run still going has drawn a tenth of the
    tokens its config will eventually put through it, and "処理トークン" is a
    statement about what happened, not about what was planned. Captioning the
    plan over a run at 3% put 430 G beside an axis that stopped at 13 G.
    """
    got = meta.get(series, {})

    def show(key, fmt="{}"):
        value = got.get(key)
        if value is None:
            return "不明"
        if isinstance(value, list):
            return f"{fmt.format(value[0])}〜{fmt.format(value[1])}"
        return fmt.format(value)

    # Tokens seen at completion -- the fifth of the five conditions, and the
    # one the x axis is now drawn in, so it has to be stated as a number too.
    seen = ""
    if per_step:
        total = got.get("max_steps")
        end = max(total) if isinstance(total, list) else total
        if end and reached and reached < end:
            seen = (f"  |  処理トークン {reached * per_step / 1e9:,.0f} G"
                    f"（予定 {end * per_step / 1e9:,.0f} G、走行中）")
        elif isinstance(total, list):
            seen = (f"  |  処理トークン {total[0] * per_step / 1e9:,.0f}〜"
                    f"{total[1] * per_step / 1e9:,.0f} G")
        elif total:
            seen = f"  |  処理トークン {total * per_step / 1e9:,.0f} G"

    line = (f"{series}  |  {n_runs} 本  |  窓 {show('window', '{:,}')}  |  "
            f"エポック {show('epochs')}  |  step {show('max_steps', '{:,}')}  |  "
            f"グローバルバッチ {show('global_batch', '{:,}')}  |  "
            f"学習率 {show('lr')}  |  評価間隔 {show('eval_every', '{:,}')}{seen}  |  "
            f"{metric}（nats/token）\n評価は {show('eval_set')}。"
            f"取り方の違う値を同じ列に置かない。")
    if many_subsets:
        # Each run is scored on its own subset's valid split, so the curves are
        # not one task measured 21 ways. Height across subsets is not a ranking;
        # the distance to that subset's own baseline is the comparable quantity.
        # A sweep is the opposite case -- one subset, one valid split -- and the
        # note would be false there, so it is attached to the figures it is true
        # of rather than to every figure with more than one curve.
        line += ("\n各 run は自分の subset の valid で採点している。"
                 "subset をまたいで高さを比べることはできない。"
                 "比べられるのは各 subset 自身の基準線からの差である。")
    return line


def draw(series, runs, metric, meta, baselines, out_path, tail_fraction=None,
         baseline_method="", collapsed=False):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(18, 9))          # 大判
    per_step = tokens_per_step(series, meta)
    last_step = max(s for pts in runs.values() for s, _ in pts)
    lo_step = last_step * (1 - tail_fraction) if tail_fraction else 0

    colours, by_arm = colour_map(runs)
    seen, ys, labelled, x_drawn = set(), [], False, []
    for subset, points in sorted(runs.items()):
        kept = [(s, v) for s, v in points if s >= lo_step]
        if not kept:
            continue
        xs = [(s * per_step if per_step else s) for s, _ in kept]
        x_drawn += [v for v in xs if v > 0]
        ys_run = [v for _, v in kept]
        colour = colours[subset]
        key = subset if by_arm else family_of(subset)
        ax.plot(xs, ys_run, color=colour, linewidth=1.3, alpha=0.8,
                label=key if key not in seen else None)
        seen.add(key)
        ys += [v for _, v in kept]
        best_step, best_value = min(kept, key=lambda p: p[1])
        best_x = best_step * per_step if per_step else best_step
        ax.plot([best_x], [best_value], marker="o", markersize=6,
                color=colour, markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        if len(runs) <= 8 or tail_fraction:
            labelled = True
            ax.annotate(f"{best_step:,} step", (best_x, best_value),
                        textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=8, color=colour)

    drawn = [baselines[k] for k in
             (s if s in baselines else base_subset(s) for s in runs)
             if k in baselines]
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
    if per_step:
        # The left edge is the first point actually drawn, not step 1. Taking
        # the schedule's start put three empty decades on a log axis, because
        # the first evaluation lands at step 1,000 -- about 2.6 G tokens, not
        # 2.6 M.
        ax.set_xscale("log")
        ax.set_xlim(min(x_drawn), last_step * per_step)
        ax.set_xlabel("処理トークン数（対数）")
    else:
        ax.set_xlim(lo_step, last_step)
        ax.set_xlabel("step")
    ax.set_ylabel(f"{metric}（nats/token）")
    if per_step:
        ax.xaxis.set_major_formatter(token_tick)
        # The minors get a label only on an axis narrower than one decade.
        # There, at most one major tick falls inside the range and matplotlib
        # labels the minors in its own notation, so the axis carries two at
        # once -- "3 x 10^9, 4 x 10^9, 6 x 10^9, 10 G". Wider than that, every
        # decade has its own major tick and labelling the minors as well runs
        # them together: "8.0 G 9.0 G10 G", "90 G100 G".
        lo_x, hi_x = ax.get_xlim()
        if lo_x > 0 and hi_x / lo_x < 10:
            ax.xaxis.set_minor_formatter(token_tick)
        else:
            ax.xaxis.set_minor_formatter(matplotlib_null_formatter())
    else:
        ax.xaxis.set_major_formatter(step_tick)
    # The collapsed runs are a separate figure; without saying so in the title
    # the two are told apart only by their file names.
    ax.set_title(("最後の 20 %: " if tail_fraction else "") + series
                 + ("（自分の基準線を上回って終わった run）" if collapsed else ""),
                 fontsize=15)
    ax.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    text = caption(series, meta, len(runs), metric,
                   len({base_subset(s) for s in runs}) > 1, per_step, last_step)
    if per_step:
        text += (f"\n横軸は処理トークン数 = step × グローバルバッチ × 窓長"
                 f"（{per_step:,} トークン/step）。計算量 C = 6ND ではない。"
                 f"6ND は系列長由来の注意機構の項を含まない。")
    fig.text(0.008, 0.012, text, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")


def draw_epoch_pair(curves, metric, meta, baselines, out_path):
    """3 epochs against 9, on the one subset that has both.

    sat9 is mammal_centered alone, so the run it is placed beside is
    mammal_centered from the 512-window series -- the same subset, the same
    window, the same batch. Any other subset would change two things at once.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(18, 9))
    # Both sides are window 512 at global batch 2,560, so one tokens-per-step
    # covers them and the axis matches every other figure here. Drawn against
    # step this was the only figure left on a different axis -- and it is the
    # one the three-versus-nine reading is made from.
    per_step = tokens_per_step("bert-base", meta) or tokens_per_step("bert-sat9", meta)
    pairs = (("bert-base", "mammal_centered", "3 エポック (bert-base)", "#1b3a6b"),
             ("bert-sat9", "mammal_centered", "9 エポック (bert-sat9)", "#c2622d"))
    ys = []
    for series, subset, label, colour in pairs:
        points = curves.get(series, {}).get(subset)
        if not points:
            print(f"  {series}/{subset} が無いので省いた")
            continue
        xs = [(st * per_step if per_step else st) for st, _ in points]
        ax.plot(xs, [v for _, v in points], color=colour, linewidth=1.6, label=label)
        ys += [v for _, v in points]
        best_step, best_value = min(points, key=lambda p: p[1])
        best_x = best_step * per_step if per_step else best_step
        ax.plot([best_x], [best_value], marker="o", markersize=7, color=colour,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        ax.annotate(f"{best_step:,} step / {best_value:.4f}", (best_x, best_value),
                    textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=10, color=colour)
    if "mammal_centered" in baselines:
        value = baselines["mammal_centered"]
        ax.axhline(value, color=BASELINE_COLOUR, linestyle="--", linewidth=1.0,
                   label=f"基準線 {value:.4f}")
        ys.append(value)
    span = max(ys) - min(ys)
    ax.set_ylim(min(ys) - span * 0.11, max(ys) + span * 0.04)
    if per_step:
        ax.set_xscale("log")
        ax.set_xlabel("処理トークン数（対数）")
        ax.xaxis.set_major_formatter(token_tick)
    else:
        ax.set_xlabel("step")
        ax.xaxis.set_major_formatter(step_tick)
    ax.set_ylabel(f"{metric}（nats/token）")
    ax.set_title("mammal_centered: 3 エポックと 9 エポック", fontsize=15)
    ax.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    note = ("窓 512・グローバルバッチ 2,560・学習率 1e-4・エポック 3 と 9・"
            "指標 eval_loss_mask（nats/token）。両者とも mammal_centered の 1 本である。")
    if per_step:
        note += (f"\n横軸は処理トークン数 = step × グローバルバッチ × 窓長"
                 f"（{per_step:,} トークン/step）。計算量 C = 6ND ではない。"
                 f"6ND は系列長由来の注意機構の項を含まない。")
    fig.text(0.008, 0.012, note, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out_path, dpi=200)
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


def per_run_table(curves, metric, meta, baselines, out_path):
    """Every run's best and final value, as a table beside the figures.

    A log axis puts 0.06 and 0.07 on top of each other at the tail, so an
    ordering the figure cannot resolve has to be readable somewhere. Best and
    final are kept in separate columns rather than mixed into one: they answer
    different questions, and a run whose best sits at 13% of its schedule is
    not described by either number alone.
    """
    import csv
    rows = []
    for series in sorted(curves):
        base = baselines.get(series, {})
        per_step = tokens_per_step(series, meta)
        for subset, points in sorted(curves[series].items()):
            points = sorted(points)
            best_step, best = min(points, key=lambda p: p[1])
            last_step, last = points[-1]
            line = base.get(subset) or base.get(base_subset(subset))
            rows.append({
                "series": series, "subset": subset, "metric": metric[series],
                "best": f"{best:.6f}", "best_step": best_step,
                "best_position_pct": f"{100 * best_step / last_step:.1f}",
                "final": f"{last:.6f}", "final_step": last_step,
                "reversal": f"{last - best:+.6f}",
                "baseline": "" if line is None else f"{line:.6f}",
                "ended_above_baseline": "" if line is None else int(last >= line),
                "tokens_seen": "" if not per_step else last_step * per_step,
            })
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {out_path}  ({len(rows)} rows)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--meta", default="")
    ap.add_argument("--baselines", nargs="*", default=[],
                    help="series=directory of <subset>.json holding degenerate_baseline")
    ap.add_argument("--baselines-tally", nargs="*", default=[],
                    help="series=path/to/degenerate-baselines.json:model")
    ap.add_argument("--skip-series", nargs="*", default=[],
                    help="series to keep in the table but not draw. Use it when a "
                         "series' labels cannot answer which subset each run is "
                         "on: colour, baseline and the comparability note all "
                         "depend on that, and a figure that guesses is worse "
                         "than a table")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="genome-loss")
    args = ap.parse_args()

    _pyplot()          # fail here, before any reading, if the stack is missing

    curves, metric = read_tsv(args.tsv)
    meta = json.load(open(args.meta)) if args.meta and os.path.exists(args.meta) else {}
    baselines, method = load_baselines(args.baselines, args.baselines_tally)
    os.makedirs(args.out_dir, exist_ok=True)

    for series in sorted(curves):
        if series in args.skip_series:
            print(f"  {series}: --skip-series により描かない（表には残る）")
            continue
        base = baselines.get(series, {})
        if not base:
            print(f"  {series}: このビルドで測った基準線が無いので引かない")
        learned, collapsed = split_collapsed(curves[series], base)
        if collapsed:
            print(f"  {series}: 崩れた {len(collapsed)} 本を別図にした "
                  f"({', '.join(sorted(collapsed))})")
        for name, runs in (("", learned), ("-collapsed", collapsed)):
            if not runs:
                continue
            is_collapsed = name == "-collapsed"
            draw(series, runs, metric[series], meta, base,
                 os.path.join(args.out_dir, f"{args.prefix}-{series}{name}.png"),
                 baseline_method=method.get(series, ""), collapsed=is_collapsed)
            draw(series, runs, metric[series], meta, base,
                 os.path.join(args.out_dir, f"{args.prefix}-{series}{name}-tail20.png"),
                 tail_fraction=0.20, baseline_method=method.get(series, ""),
                 collapsed=is_collapsed)

    table = per_run_table(curves, metric, meta, baselines,
                          os.path.join(args.out_dir, f"{args.prefix}-per-run.tsv"))
    above = sum(1 for r in table if r["ended_above_baseline"] == 1)
    print(f"  自分の基準線を上回って終わった run: {above} / {len(table)}")

    draw_epoch_pair(curves, "eval_loss_mask", meta, baselines.get("bert-base", {}),
                    os.path.join(args.out_dir, f"{args.prefix}-epoch-3-vs-9.png"))


if __name__ == "__main__":
    main()
