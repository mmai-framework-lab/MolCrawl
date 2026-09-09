"""Model-free baselines for the zero-shot ClinVar score.

The composition baseline built earlier is not the right comparison for zero-shot
and is being kept for the linear probe instead: it fits a rate to the labels, so
it knows which substitutions are pathogenic. Zero-shot never sees a label. It
only says how likely a base is at a position, so the line it has to clear must
be drawn by something that also never sees a label.

Two are computed, both scoring a variant exactly the way the models are scored
-- reference minus variant, higher meaning "the reference was the likelier
sequence":

  unigram   log f(alt) - log f(ref), f the base frequency of the scored
            sequence. Knows nothing but how common each base is.

  k-mer     the difference of whole-window log likelihood under an order-(k-1)
            Markov model. Knows the local statistics of the genome and nothing
            else, which is what makes it the interesting line: a pretrained model
            that does not beat it has not learned more than the shared statistics
            any counting program can collect.

The Markov model is fitted on the sequences being scored. That is deliberately
generous -- it cannot be surprised by the very windows it is tested on -- so
failing to beat it is a strong statement and beating it is a weak one.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import Counter

BASES = "ACGT"


def read_rows(path, chroms=None):
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    rows = []
    for r in csv.DictReader(open(path, newline="")):
        c = (r.get("chrom") or "").replace("chr", "").strip()
        if chroms and c not in chroms:
            continue
        rows.append(r)
    return rows


def unigram_scores(rows):
    """log f(alt) - log f(ref): the whole model is four numbers."""
    counts = Counter()
    for r in rows:
        counts.update(b for b in r["reference_sequence"] if b in BASES)
    total = sum(counts.values())
    logf = {b: math.log(counts[b] / total) for b in BASES if counts[b]}
    out = []
    for r in rows:
        ref, alt = r["ref"].upper(), r["alt"].upper()
        if ref in logf and alt in logf:
            # Same orientation as the models: reference likelier => higher score.
            out.append(logf[ref] - logf[alt])
        else:
            out.append(0.0)
    return out, {b: counts[b] / total for b in BASES if counts[b]}


def fit_markov(rows, k):
    """Order-(k-1) transition counts over the reference windows."""
    ctx = Counter()
    nxt = Counter()
    order = k - 1
    for r in rows:
        s = r["reference_sequence"]
        for i in range(order, len(s)):
            c = s[i - order:i]
            b = s[i]
            if b in BASES and all(x in BASES for x in c):
                ctx[c] += 1
                nxt[(c, b)] += 1
    return ctx, nxt, order


def markov_loglik(seq, ctx, nxt, order):
    """Log likelihood of a window, positions with enough context only.

    Both sequences of a pair are scored over the same positions, so the constant
    offset from skipping the first `order` bases cancels in the difference.
    """
    ll = 0.0
    for i in range(order, len(seq)):
        c, b = seq[i - order:i], seq[i]
        if b not in BASES or not all(x in BASES for x in c):
            continue
        # add-one over the four bases: an unseen context is uninformative rather
        # than infinitely unlikely
        ll += math.log((nxt[(c, b)] + 1.0) / (ctx[c] + 4.0))
    return ll


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinvar-csv", required=True)
    ap.add_argument("--chroms", default="21,22,X,Y")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import numpy as np

    chroms = {c.strip() for c in args.chroms.split(",") if c.strip()}
    rows = read_rows(args.clinvar_csv, chroms)
    labels = [1 if "pathogenic" in (r["ClinicalSignificance"] or "").lower()
              and "conflicting" not in (r["ClinicalSignificance"] or "").lower()
              else 0 for r in rows]
    win = len(rows[0]["reference_sequence"])
    print(f"  variants {len(rows):,}   window {win}   "
          f"pathogenic {sum(labels):,}   benign {len(labels) - sum(labels):,}")

    uni, freq = unigram_scores(rows)
    a_uni = auroc(labels, uni)
    print(f"\n  塩基頻度 {', '.join(f'{b} {freq[b]:.3f}' for b in sorted(freq))}")
    print(f"  1. unigram  log f(ref) - log f(alt)      AUROC {a_uni:.4f}")

    ctx, nxt, order = fit_markov(rows, args.k)
    mk = [markov_loglik(r["reference_sequence"], ctx, nxt, order)
          - markov_loglik(r["variant_sequence"], ctx, nxt, order) for r in rows]
    a_mk = auroc(labels, mk)
    print(f"  2. {args.k}-mer Markov (order {order}) 窓全体の対数尤度差  "
          f"AUROC {a_mk:.4f}   文脈 {len(ctx):,} 種")

    out = {"variants": len(rows), "window": win, "k": args.k,
           "base_frequency": freq,
           "unigram_auroc": a_uni, "markov_auroc": a_mk,
           "per_chrom": {}}
    print("\n  === 染色体ごと ===")
    lab = np.asarray(labels)
    ch = np.array([(r.get("chrom") or "").replace("chr", "").strip() for r in rows])
    for c in sorted(set(ch.tolist())):
        sel = ch == c
        n = int(sel.sum())
        u = auroc(lab[sel], np.asarray(uni)[sel])
        m = auroc(lab[sel], np.asarray(mk)[sel])
        note = "  ← 29 件、合計に含めない" if n < 100 else ""
        print(f"  chr{c:<3s} {n:>6,} 件   unigram {u:.4f}   markov {m:.4f}{note}")
        out["per_chrom"][c] = {"n": n, "unigram_auroc": u, "markov_auroc": m}

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
