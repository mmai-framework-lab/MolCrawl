"""How many ClinVar variants sit on the chromosomes no subset was trained on.

The upstream evaluation is scored on chr21 / 22 / X / Y, which every one of the
21 pretraining subsets held out. Restricting it there is what keeps
mammal_centered comparable: that subset alone contains the human genome, and it
learned chr1 through chr20, so a genome-wide evaluation would score one run of
twenty-one on sequence it had already seen.

The cost of the restriction is sample size, and that is what this counts. The
figure that matters is not the total but the smaller of the two classes, since a
balanced comparison is bounded by it.

Counts are reported per split as well, because a fine-tune that trains on the
same variants it is scored on measures nothing; splits_manifest.csv is the
authority on which variant went where. It carries two rows per variant -- the
reference sequence and the altered one -- so a variant is counted once here, off
the ``ref`` row, rather than twice.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

# The four the pretraining subsets hold out, as they appear in the CSV.
HELDOUT = ("21", "22", "X", "Y")

# ClinVar's ClinicalSignificance is free text with combinations and qualifiers.
# Collapse to the two classes the evaluation uses; anything else is neither, and
# is reported separately rather than being forced into one of them.
PATHOGENIC = ("pathogenic",)
BENIGN = ("benign",)


def classify(sig):
    s = (sig or "").strip().lower()
    if not s:
        return "unclassified"
    # "conflicting" carries both words and belongs to neither class.
    if "conflicting" in s:
        return "conflicting"
    has_p = any(k in s for k in PATHOGENIC)
    has_b = any(k in s for k in BENIGN)
    if has_p and has_b:
        return "conflicting"
    if has_p:
        return "pathogenic"
    if has_b:
        return "benign"
    return "other"


def wilson(k, n, z=1.96):
    """Interval for a proportion at small n, where the normal one misbehaves."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True, help="clinvar_sequences.csv")
    ap.add_argument("--splits", default="", help="splits_manifest.csv")
    ap.add_argument("--chroms", default=",".join(HELDOUT))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    want = [c.strip() for c in args.chroms.split(",") if c.strip()]
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    per_chrom = defaultdict(Counter)
    overall = Counter()
    vcv_class = {}
    with open(args.sequences, newline="") as fh:
        for row in csv.DictReader(fh):
            cls = classify(row.get("ClinicalSignificance"))
            chrom = (row.get("chrom") or "").replace("chr", "").strip()
            overall[cls] += 1
            if chrom in want:
                per_chrom[chrom][cls] += 1
                vcv_class[row.get("vcv_id")] = (chrom, cls)

    print("  === 保留染色体ごと ===")
    print(f"  {'chrom':>6s} {'病原性':>9s} {'良性':>9s} {'競合':>8s} {'その他':>8s} {'計':>9s}")
    tot = Counter()
    for c in want:
        d = per_chrom[c]
        tot.update(d)
        print(f"  {c:>6s} {d['pathogenic']:>9,} {d['benign']:>9,}"
              f" {d['conflicting']:>8,} {d['other']+d['unclassified']:>8,}"
              f" {sum(d.values()):>9,}")
    n_p, n_b = tot["pathogenic"], tot["benign"]
    print(f"  {'計':>6s} {n_p:>9,} {n_b:>9,} {tot['conflicting']:>8,}"
          f" {tot['other']+tot['unclassified']:>8,} {sum(tot.values()):>9,}")

    smaller = min(n_p, n_b)
    print(f"\n  2 クラスの小さい方: {smaller:,}"
          f"  ({'500 件以上' if smaller >= 500 else '500 件未満 — 信頼区間を併記'})")
    if n_p + n_b:
        lo, hi = wilson(n_p, n_p + n_b)
        print(f"  病原性の割合 {n_p/(n_p+n_b):.3f}  95% 区間 [{lo:.3f}, {hi:.3f}]")
        # What a balanced comparison can resolve at this size.
        half = wilson(smaller, 2 * smaller)
        if half:
            print(f"  均衡 {smaller:,}+{smaller:,} での精度の 95% 区間幅"
                  f" ≈ ±{(half[1]-half[0])/2:.3f}")

    splits = {}
    if args.splits and os.path.exists(args.splits):
        per_split = defaultdict(Counter)
        with open(args.splits, newline="") as fh:
            for row in csv.DictReader(fh):
                # Two rows per variant (kind=ref and kind=var, same vcv_id and
                # same split). One variant, one count.
                if (row.get("kind") or "").strip() != "ref":
                    continue
                hit = vcv_class.get(row.get("vcv_id"))
                if hit:
                    per_split[row["split"]][hit[1]] += 1
        print("\n  === split ごと（保留染色体のみ、変異 1 件 = 1 カウント） ===")
        for sp in sorted(per_split):
            d = per_split[sp]
            print(f"  {sp:>6s} 病原性 {d['pathogenic']:>7,}  良性 {d['benign']:>7,}"
                  f"  小さい方 {min(d['pathogenic'], d['benign']):>7,}")
        splits = {k: dict(v) for k, v in per_split.items()}

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"chroms": want,
                   "per_chrom": {k: dict(v) for k, v in per_chrom.items()},
                   "heldout_total": dict(tot),
                   "genome_wide_total": dict(overall),
                   "per_split": splits}, open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
