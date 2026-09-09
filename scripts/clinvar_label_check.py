"""Check the ClinVar labels against the variant's own consequence annotation.

This is a sanity check on the data, not a model result. `consequence` says what a
variant does to the transcript -- nonsense, frameshift, missense, synonymous --
and those categories carry an obvious prior: a stop codon or a frame shift breaks
the protein, a synonymous change usually does not. So scoring the label with
`consequence` as the only feature has to come out well above 0.5.

If it lands near 0.5, the labels and the windows have come apart somewhere -- a
join gone wrong, a shifted row, a mislabelled column -- and no model number
computed on this table means anything. That is what makes this worth running
before reading any AUROC.

No model, no GPU: it is a contingency table.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter


def auroc(labels, scores):
    import numpy as np
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
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


def label_of(sig):
    s = (sig or "").lower()
    if "conflicting" in s:
        return None
    p, b = "pathogenic" in s, "benign" in s
    if p and b:
        return None
    return 1 if p else (0 if b else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinvar-csv", required=True)
    ap.add_argument("--chroms", default="21,22,X,Y")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    chroms = {c.strip() for c in args.chroms.split(",") if c.strip()}

    cons, labels = [], []
    for r in csv.DictReader(open(args.clinvar_csv, newline="")):
        c = (r.get("chrom") or "").replace("chr", "").strip()
        if chroms and c not in chroms:
            continue
        y = label_of(r.get("ClinicalSignificance"))
        if y is None:
            continue
        cons.append((r.get("consequence") or "").strip() or "(none)")
        labels.append(y)

    n_pos = sum(labels)
    print(f"  variants {len(labels):,}   pathogenic {n_pos:,}   benign {len(labels) - n_pos:,}")

    # Score each variant by the pathogenic rate of its own consequence class.
    # Fitted on the same rows, which is the point: if even that cannot separate
    # the labels, the labels are not attached to what they should be.
    tot, pos = Counter(cons), Counter()
    for c, y in zip(cons, labels):
        if y:
            pos[c] += 1
    rate = {c: (pos[c] + 0.5) / (tot[c] + 1.0) for c in tot}
    a = auroc(labels, [rate[c] for c in cons])

    print(f"\n  consequence だけを score にした AUROC: {a:.4f}")
    verdict = ("ラベルは consequence と整合しています" if a >= 0.7 else
               "0.7 未満。窓かラベルの対応を確認してください" if a >= 0.55 else
               "0.5 付近。ラベルの割り当てか窓の対応が壊れています")
    print(f"  → {verdict}")

    print(f"\n  {'consequence':44s} {'件数':>8s} {'病原性':>8s} {'率':>7s}")
    for c, n in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"  {c[:44]:44s} {n:>8,} {pos[c]:>8,} {pos[c]/n:>6.1%}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"variants": len(labels), "pathogenic": n_pos, "auroc": a,
                   "per_consequence": {c: {"n": tot[c], "pathogenic": pos[c],
                                           "rate": pos[c] / tot[c]} for c in tot}},
                  open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
