"""The schedule measurement: what it showed, and why one run cannot settle it.

Two figures and the table behind them.

**The relationship first.** The measurement asks whether the nine-epoch runs'
reversal comes from tripling the high-learning-rate stretch or from the three
passes over the data. That question only has an answer on a run that reversed.
Figure 1 pairs each subset's reversal at three epochs with its reversal at nine,
so where mammal_centered sits among the twenty-one is the first thing seen --
and it is the one subset of twenty-one whose nine-epoch run did not reverse.

**The curves second.** Figure 2 is the three schedules on mammal_centered. It is
an appendix figure: three curves separated by less than a tenth of the gap to the
model-free floor, which a reader cannot resolve by eye. The numbers are written
to a table beside it for that reason.

No ordering is claimed between the three. The evaluation redraws 64,000 rows with
replacement every time, adjacent evaluation points differ by 0.0019 at the median,
and the gaps here are of that size. Seed variance was not measured for these.
"""
import argparse
import csv
import os
from collections import defaultdict

FAMILY_COLOUR = {"mammal_centered": "#1b3a6b",
                 "eukaryote_matched": "#c2622d",
                 "global_random": "#3f8f6f"}
HIGHLIGHT = "#1b3a6b"
BASELINE_COLOUR = "#8a8a8a"
SCHEDULE_COLOUR = {"gpt2": "#8a6d1f",
                   "gpt2-ep9": "#c2622d",
                   "gpt2-ep9-decay3": "#1b3a6b"}
SCHEDULE_LABEL = {"gpt2": "3 エポック（cosine 54,634）",
                  "gpt2-ep9": "9 エポック（cosine 163,900）",
                  "gpt2-ep9-decay3": "9 エポック / cosine は 3 エポック分（54,634）"}


def _pyplot():
    """Imported here so the rules above can be tested without matplotlib.

    The order is load-bearing, and the same as plot_genome_loss_curves.py: the
    backend is selected before pyplot is imported, and japanize_matplotlib sets
    the font family as a side effect of its own import. Left out, every
    Japanese character on the figure renders as an empty box -- which is how
    the first version of these two figures came out.
    """
    import matplotlib
    matplotlib.use("Agg")
    import japanize_matplotlib  # noqa: F401  (no CJK font is installed system-wide)
    import matplotlib.pyplot as plt
    # japanize_matplotlib set the family; only the minus sign is left, which
    # IPAexGothic renders as a full-width dash without this.
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def family_of(subset):
    if subset.startswith("mammal"):
        return "mammal_centered"
    if subset.startswith("eukaryote"):
        return "eukaryote_matched"
    return "global_random"


def read_per_run(path, series):
    """{subset: row} for one series of the per-run table."""
    out = {}
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["series"] == series:
                out[row["subset"]] = row
    return out


def read_curves(path, series, split="validation"):
    """{subset: [(step, value)]} for one series and one split.

    The table carries both splits at the same steps, so without the filter each
    curve is drawn twice.
    """
    out = defaultdict(list)
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["series"] in series and row.get("split", "validation") == split:
                out[(row["series"], row["subset"])].append(
                    (int(row["step"]), float(row["value"])))
    return {k: sorted(v) for k, v in out.items()}


def outlier_gap(nine):
    """The smallest reversal, and how far the next one is above it.

    Recomputed from the table every time rather than carried over from the
    previous report -- "15 collapsed" survived three revisions after it had
    become 25.
    """
    ranked = sorted(nine.items(), key=lambda kv: float(kv[1]["reversal"]))
    first, second = ranked[0], ranked[1]
    return (first[0], float(first[1]["reversal"]),
            second[0], float(second[1]["reversal"]), len(ranked))


def draw_pairs(three, nine, out_path):
    """Figure 1: each subset's reversal at three epochs against at nine."""
    plt = _pyplot()
    shared = sorted(set(three) & set(nine))
    fig, ax = plt.subplots(figsize=(18, 9))
    for subset in shared:
        a, b = float(three[subset]["reversal"]), float(nine[subset]["reversal"])
        highlight = subset == "mammal_centered"
        ax.plot([0, 1], [a, b],
                color=HIGHLIGHT if highlight else FAMILY_COLOUR[family_of(subset)],
                linewidth=3.0 if highlight else 1.2,
                alpha=1.0 if highlight else 0.55,
                marker="o", markersize=9 if highlight else 5, zorder=3 if highlight else 2)
        if highlight:
            ax.annotate(f"{subset}  {a:+.4f} → {b:+.4f}", (1, b),
                        textcoords="offset points", xytext=(14, 0), va="center",
                        fontsize=13, color=HIGHLIGHT, fontweight="bold")
    ax.set_xlim(-0.08, 0.62)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["3 エポック", "9 エポック"], fontsize=13)
    ax.set_yscale("log")
    ax.set_ylabel("反転 = 最終 val_loss − 最良 val_loss（nats/token、対数）")
    ax.grid(alpha=0.25, linewidth=0.5, axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    name, low, name2, second, n = outlier_gap(nine)
    ax.set_title(f"9 エポックで反転しなかったのは {n} 本中 {name} の 1 本だけである", fontsize=15)
    handles = [plt.Line2D([], [], color=c, linewidth=2, marker="o", label=k)
               for k, c in FAMILY_COLOUR.items()]
    ax.legend(handles=handles, fontsize=11, loc="upper left", framealpha=0.9)
    fig.text(0.01, 0.015,
             f"GPT-2 small、subset ごとに 1 本、{n} subset を対で結んだ。"
             f"縦軸は各 run の最終 val_loss から最良 val_loss を引いた値。"
             f"\n{name} は 9 エポックで {low:+.4f}、次に小さい {name2} は {second:+.4f} で、"
             f"{second / low:.0f} 倍の開きがある。"
             f"\n評価は valid から毎回抽選 64,000 行（重複あり）。"
             f"隣り合う評価点どうしの差は中央値 0.0019。seed のばらつきは測っていない。",
             fontsize=11, va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")


def draw_schedules(curves, baseline, per_step, out_path, tail=None):
    """Figure 2: the three schedules on one subset, against processed tokens."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(18, 9))
    ys = []
    for (series, _subset), points in sorted(curves.items()):
        xs = [s * per_step for s, _ in points if s > 0]
        vs = [v for s, v in points if s > 0]
        ax.plot(xs, vs, color=SCHEDULE_COLOUR[series], linewidth=1.6,
                label=SCHEDULE_LABEL[series])
        ys += vs
    if baseline is not None and not tail:
        ax.axhline(baseline, color=BASELINE_COLOUR, linestyle="--", linewidth=1.0,
                   label=f"基準線 {baseline:.4f}（採点と同一パス）")
        ys.append(baseline)
    ax.set_ylim(*band_limits(ys, tail))
    ax.set_xscale("log")
    ax.set_xlabel("処理トークン数（対数）")
    ax.set_ylabel("val_loss（nats/token）")
    ax.xaxis.set_major_formatter(_token_tick)
    ax.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.set_title("mammal_centered: 学習率日程を 3 通りにした同一 subset の 3 本"
                 + (f"（縦軸を {tail:.2f} までに切った拡大）" if tail else ""),
                 fontsize=15)
    fig.text(0.01, 0.015,
             "GPT-2 small、subset は mammal_centered、seed 45、学習率 1e-4、min_lr 1e-5。"
             "グローバルバッチ 2,560 | 窓 1,024 | 評価間隔 1,000。"
             "\n処理トークン数 = step × グローバルバッチ × 窓長（2,621,440 トークン/step）。"
             "3 エポックは 54,634 step・143 G、9 エポックと decay3 は 163,900 step・430 G。"
             "\n評価は valid から毎回抽選 64,000 行（重複あり）。"
             "隣り合う評価点どうしの差は中央値 0.0019 で、3 本の差はその程度しかない。"
             "順位は主張しない（seed のばらつき未測定）。数値は添付の表で読む。"
             + ("\n拡大図は縦軸を切っているだけで、点は捨てていない。"
                "帯の外へ出た曲線は視野の外へ出る。" if tail else ""),
             fontsize=11, va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")


def band_limits(values, tail=None):
    """The y range to show, given every drawn value.

    With a band, the axis is clipped to it and every point is still plotted.
    Dropping the points above the band and drawing on leaves the line joined
    across the gap, so a spike that left the band and came back appears as a
    straight climb that never happened -- which is how the first version of the
    zoomed figure came out, with a rise of 0.018 across the middle of it.
    """
    if tail is None:
        span = max(values) - min(values)
        return min(values) - span * 0.06, max(values) + span * 0.06
    inside = [v for v in values if v <= tail]
    span = tail - min(inside)
    return min(inside) - span * 0.06, tail


def _token_tick(value, _pos=None):
    if value >= 1e10:
        return f"{value / 1e9:,.0f} G"
    if value >= 1e9:
        return f"{value / 1e9:.1f} G"
    return f"{value / 1e6:,.0f} M"


def write_table(curves, per_step, out_path):
    """The numbers the log axis cannot resolve, as a table."""
    rows = []
    for (series, subset), points in sorted(curves.items()):
        best_step, best = min(points, key=lambda sv: sv[1])
        last_step, last = points[-1]
        rows.append({
            "series": series, "subset": subset, "schedule": SCHEDULE_LABEL[series],
            "eval_points": len(points), "last_step": last_step,
            "tokens_seen": last_step * per_step,
            "best": f"{best:.4f}", "best_step": best_step,
            "best_position_pct": f"{100 * best_step / last_step:.1f}",
            "final": f"{last:.4f}", "reversal": f"{last - best:+.4f}"})
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {out_path}  ({len(rows)} rows)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="long-form table, one row per eval point")
    ap.add_argument("--per-run", required=True, help="one row per run")
    ap.add_argument("--subset", default="mammal_centered")
    ap.add_argument("--baseline", type=float, default=None,
                    help="the model-free value for this subset's build")
    ap.add_argument("--per-step", type=int, default=2621440)
    ap.add_argument("--tail-below", type=float, default=1.12,
                    help="the zoomed figure keeps points at or below this")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    three = read_per_run(args.per_run, "gpt2")
    nine = read_per_run(args.per_run, "gpt2-ep9")
    name, low, name2, second, n = outlier_gap(nine)
    print(f"  9 エポック {n} 本: 反転が最小なのは {name} {low:+.4f}、"
          f"次が {name2} {second:+.4f}（{second / low:.0f} 倍）")

    draw_pairs(three, nine, os.path.join(args.out_dir, "genome-schedule-reversal-pairs.png"))

    series = ("gpt2", "gpt2-ep9", "gpt2-ep9-decay3")
    curves = {k: v for k, v in read_curves(args.tsv, series).items()
              if k[1] == args.subset}
    missing = [s for s in series if (s, args.subset) not in curves]
    if missing:
        raise SystemExit(f"{args.subset} の曲線が無い系列: {missing}")
    write_table(curves, args.per_step,
                os.path.join(args.out_dir, "genome-schedule-measurement.tsv"))
    draw_schedules(curves, args.baseline, args.per_step,
                   os.path.join(args.out_dir, "genome-schedule-measurement.png"))
    draw_schedules(curves, args.baseline, args.per_step,
                   os.path.join(args.out_dir, "genome-schedule-measurement-zoom.png"),
                   tail=args.tail_below)


if __name__ == "__main__":
    main()
