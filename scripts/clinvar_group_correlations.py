"""Within-group correlations between a subset's shape and its ClinVar AUROC.

Three candidate explanations for why `global_random` outscores
`eukaryote_matched` are checkable without training anything: how many
assemblies a subset drew, how long its contigs are, and how much of it was lost
to windowing.

**The correlation is taken inside each group, never across the two.** Assembly
count is 41-112 for `eukaryote_matched` and 473-885 for `global_random` -- the
ranges do not overlap, so the variable is almost a label for the group. Pooled,
any group difference reappears as a correlation with it, and the result says
only that the two groups differ, which was the question rather than the answer.
If the relationship does not hold inside a group, assembly count does not
explain anything.

`mammal_centered` is one subset. One point has no correlation, so it is not a
group here and is left out rather than folded into either of the other two.

Both coefficients are reported. Pearson asks whether the relationship is linear
and Spearman whether it is monotone; where they disagree in sign, that is
printed rather than resolved, because a disagreement at n=10 is itself the
finding. At n=10 a two-sided p of 0.05 needs |r| around 0.63.
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

GROUPS = ("global_random", "eukaryote_matched")
FOLDS = ("21", "22", "X")


def _has_spread(deviations, values):
    """Whether a column varies enough for a correlation to mean anything.

    Testing the sum of squares against zero is not enough. Ten copies of 0.70
    average to 0.6999999999999998, so the deviations are about 1e-17 rather
    than 0 and the sum of squares is about 1e-33 -- truthy. The division then
    returns a number built entirely out of rounding error. The comparison has
    to be against the scale of the values, not against zero.
    """
    scale = max((abs(v) for v in values), default=0.0) or 1.0
    return math.sqrt(sum(d * d for d in deviations) / len(deviations)) > 1e-12 * scale


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    dx = [a - mx for a in xs]
    dy = [b - my for b in ys]
    if not _has_spread(dx, xs) or not _has_spread(dy, ys):
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(
        sum(a * a for a in dx) * sum(b * b for b in dy))


def _ranks(values):
    """Average ranks, so ties do not depend on the order they arrived in."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs, ys):
    return pearson(_ranks(xs), _ranks(ys))


def p_two_sided(r, n):
    """Fisher's z. At these sizes it is an approximation and is labelled one."""
    if r is None or n < 4 or abs(r) >= 1:
        return None
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(n - 3)
    return math.erfc(abs(z) / math.sqrt(2))


def family_of(subset):
    if subset.startswith("mammal_centered"):
        return "mammal_centered"
    return "eukaryote_matched" if subset.startswith("eukaryote_matched") else "global_random"


def assembly_counts(subsets_dir):
    """Rows per subset CSV -- one row is one assembly the subset drew."""
    out = {}
    for path in sorted(glob.glob(os.path.join(subsets_dir, "*.csv"))):
        with open(path) as handle:
            out[os.path.splitext(os.path.basename(path))[0]] = sum(1 for _ in handle) - 1
    return out


def window_shape(window_loss_path):
    """Mean contig length and the fraction of bases windowing threw away."""
    rows = json.load(open(window_loss_path))
    return ({r["subset"]: r["mean_segment_len"] for r in rows},
            {r["subset"]: r["loss_fraction"] for r in rows})


def read_auroc(results_tsv):
    """{(arch, method, fold): {subset: auroc}} for the trained runs only."""
    out = defaultdict(dict)
    with open(results_tsv) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["kind"] != "run":
                continue
            out[(row["arch"], row["method"], row["fold"])][row["subset"]] = float(row["auroc"])
    return out


def correlate(auroc, axes):
    rows = []
    for (arch, method, fold), by_subset in sorted(auroc.items()):
        for axis_name, axis in axes.items():
            for group in GROUPS:
                subsets = sorted(s for s in by_subset
                                 if family_of(s) == group and s in axis)
                xs = [axis[s] for s in subsets]
                ys = [by_subset[s] for s in subsets]
                r_p, r_s = pearson(xs, ys), spearman(xs, ys)
                rows.append({
                    "arch": arch, "method": method, "fold": fold, "group": group,
                    "axis": axis_name, "n": len(subsets),
                    "pearson": r_p, "spearman": r_s,
                    "p_pearson": p_two_sided(r_p, len(subsets)),
                    "p_spearman": p_two_sided(r_s, len(subsets)),
                    "signs_agree": None if r_p is None or r_s is None else (r_p * r_s > 0),
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-tsv", required=True)
    ap.add_argument("--subsets-dir", required=True, help="assets/genome_species_list/subsets")
    ap.add_argument("--window-loss", required=True)
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    counts = assembly_counts(args.subsets_dir)
    seg_len, loss = window_shape(args.window_loss)
    axes = {"assemblies": counts, "mean_segment_len": seg_len, "window_loss_fraction": loss}
    for name, axis in axes.items():
        spread = {g: sorted(v for s, v in axis.items() if family_of(s) == g) for g in GROUPS}
        overlap = (min(spread[GROUPS[0]]) <= max(spread[GROUPS[1]])
                   and min(spread[GROUPS[1]]) <= max(spread[GROUPS[0]]))
        print(f"  {name:22s} global {min(spread['global_random']):.4g}"
              f"〜{max(spread['global_random']):.4g}   "
              f"eukaryote {min(spread['eukaryote_matched']):.4g}"
              f"〜{max(spread['eukaryote_matched']):.4g}"
              f"   群の範囲が重なる: {'はい' if overlap else '**いいえ**'}")

    rows = correlate(read_auroc(args.results_tsv), axes)
    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    with open(args.out_tsv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  wrote {args.out_tsv}  ({len(rows)} rows)")

    strong = [r for r in rows if r["pearson"] is not None and abs(r["pearson"]) >= 0.63]
    split = [r for r in rows if r["signs_agree"] is False]
    print(f"  |Pearson| >= 0.63 (n=10 で p≈0.05): {len(strong)} / {len(rows)}")
    print(f"  Pearson と Spearman で符号が割れた: {len(split)} / {len(rows)}")

    if args.out_md:
        lines = ["# subset の形と ClinVar AUROC の群内相関", "",
                 "各群の 10 本の中だけで取っている。群をまたいでいない。",
                 "n = 10 で両側 p = 0.05 に要る |r| は約 0.63 である。", ""]
        for axis_name in axes:
            lines += [f"## {axis_name}", "",
                      "| arch | 手法 | fold | 群 | n | Pearson | Spearman | 符号 |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in rows:
                if r["axis"] != axis_name:
                    continue
                mark = "" if r["signs_agree"] else " **割れ**"
                lines.append(
                    f"| {r['arch']} | {r['method']} | chr{r['fold']} | {r['group']} | "
                    f"{r['n']} | {r['pearson']:+.3f} | {r['spearman']:+.3f} |{mark or ' 一致'} |")
            lines.append("")
        open(args.out_md, "w").write("\n".join(lines))
        print(f"  wrote {args.out_md}")


if __name__ == "__main__":
    main()
