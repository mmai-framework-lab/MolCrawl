"""Aggregate per-variant ClinVar predictions across a subset campaign.

Every run scores the same variants, so the comparison between subsets is a
paired one and does not have to go through the absolute numbers. That matters
here: on the held-out chromosomes a balanced comparison of 1,646 variants gives
an accuracy interval of about +/-0.024, far wider than the differences between
subsets, while the paired difference is bounded by how much the two runs
disagree on the same variant rather than by the size of the test set.

Intervals are bootstrapped over variants, not over runs. There is one run per
subset, so run-to-run spread is not measured here and cannot be; what is
measured is how much of a difference survives resampling the variants it was
computed on. Two subsets whose intervals overlap are not ordered by this.

The degenerate baseline is the AUROC of predicting from base composition alone.
A model that has learned nothing about context still scores above 0.5 if
pathogenic and benign variants differ in which bases they involve, so the line a
result has to clear is not 0.5.
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np


def load_run(path):
    """One run's predictions, ordered by vcv_id so runs line up variant for variant."""
    rows = [json.loads(line) for line in open(path)]
    rows.sort(key=lambda r: r["vcv_id"])
    return rows


def auroc(labels, scores):
    """Rank-based AUROC, ties averaged. No threshold is involved."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # average ranks within ties so equal scores cannot favour either class
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def boot_indices(n, rounds, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(rounds, n))


def composition_baseline(rows):
    """AUROC obtainable from which bases the variant involves, nothing else.

    Scores each variant by the log-odds of being pathogenic given its
    (ref, alt) pair, estimated on the very variants being scored. That is
    optimistic by construction, which is what makes it a line to clear.
    """
    pair = [(r["ref_allele"], r["alt_allele"]) for r in rows]
    y = np.array([r["label_pathogenic"] for r in rows])
    n_pos = Counter(p for p, lab in zip(pair, y) if lab == 1)
    n_all = Counter(pair)
    rate = {p: (n_pos[p] + 0.5) / (n_all[p] + 1.0) for p in n_all}
    return auroc(y, [rate[p] for p in pair])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1026)
    ap.add_argument("--reference", default="",
                    help="subset every other one is compared against; default the median run")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.scores_dir, "*", "predictions.jsonl")))
    if not files:
        raise SystemExit(f"{args.scores_dir}: no predictions.jsonl under it")

    runs = {}
    order = None
    for f in files:
        s = os.path.basename(os.path.dirname(f))
        rows = load_run(f)
        ids = [r["vcv_id"] for r in rows]
        if order is None:
            order, labels, chrom = ids, np.array([r["label_pathogenic"] for r in rows]), \
                                   np.array([str(r["chrom"]) for r in rows])
            base_rows = rows
        elif ids != order:
            raise SystemExit(f"{s}: scores a different variant set; a paired comparison "
                             f"needs every run on the same variants")
        runs[s] = np.array([r["score"] for r in rows], dtype=float)

    n = len(order)
    print(f"  runs {len(runs)}   variants {n:,}   "
          f"pathogenic {int(labels.sum()):,}   benign {int((labels == 0).sum()):,}")

    base = composition_baseline(base_rows)
    print(f"  degenerate baseline (base composition alone): AUROC {base:.4f}\n")

    idx = boot_indices(n, args.rounds, args.seed)

    point = {s: auroc(labels, v) for s, v in runs.items()}
    ranked = sorted(point, key=lambda s: -point[s])
    ref = args.reference or ranked[len(ranked) // 2]

    # Per-run interval, and the paired difference against the reference run.
    stats = {}
    for s, v in runs.items():
        boots = np.array([auroc(labels[i], v[i]) for i in idx])
        d = np.array([auroc(labels[i], v[i]) - auroc(labels[i], runs[ref][i]) for i in idx])
        stats[s] = {
            "auroc": point[s],
            "auroc_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "margin_over_baseline": point[s] - base,
            "paired_diff_vs_ref": point[s] - point[ref],
            "paired_diff_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        }

    print(f"  reference run: {ref}\n")
    print(f"  {'subset':36s} {'AUROC':>7s} {'95% 区間':>17s} {'基準線との差':>10s}"
          f" {'対 ref':>8s} {'差の95%区間':>17s} 区別")
    for s in ranked:
        t = stats[s]
        lo, hi = t["paired_diff_ci"]
        sep = "" if (lo <= 0 <= hi) else "*"
        print(f"  {s:36s} {t['auroc']:7.4f} [{t['auroc_ci'][0]:.4f},{t['auroc_ci'][1]:.4f}]"
              f" {t['margin_over_baseline']:+10.4f} {t['paired_diff_vs_ref']:+8.4f}"
              f" [{lo:+.4f},{hi:+.4f}] {sep}")

    sep_n = sum(1 for s in ranked if s != ref and not
                (stats[s]["paired_diff_ci"][0] <= 0 <= stats[s]["paired_diff_ci"][1]))
    print(f"\n  reference と区別できる subset: {sep_n} / {len(ranked) - 1}")
    print(f"  値の幅 {point[ranked[-1]]:.4f} 〜 {point[ranked[0]]:.4f}"
          f" = {point[ranked[0]] - point[ranked[-1]]:.4f}")

    # Per chromosome. chrY is reported but never folded into a total: 29 variants
    # is not a measurement, and hiding it inside a sum would let it look like one.
    print("\n  === 染色体ごと ===")
    for c in sorted(set(chrom.tolist())):
        sel = chrom == c
        npos, nneg = int(labels[sel].sum()), int((labels[sel] == 0).sum())
        vals = {s: auroc(labels[sel], v[sel]) for s, v in runs.items()}
        note = "  ← 29 件、合計には含めない" if npos + nneg < 100 else ""
        print(f"  chr{c:<3s} 病原性 {npos:>6,} 良性 {nneg:>6,}   "
              f"AUROC {min(vals.values()):.4f} 〜 {max(vals.values()):.4f}{note}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"variants": n, "reference": ref, "degenerate_baseline": base,
                   "bootstrap_rounds": args.rounds, "seed": args.seed,
                   "per_subset": stats}, open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
