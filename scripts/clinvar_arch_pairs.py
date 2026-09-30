"""Paired differences between two ways of scoring the same ClinVar variants.

Three comparisons the campaign needs and none of the per-architecture files can
answer: BERT zero-shot against BERT probe, GPT-2 zero-shot against BERT
zero-shot, and GPT-2 probe against BERT probe.

**These are paired, and the pairing is what makes them readable.** Both sides
score the same variants, so the difference can be taken variant by variant and
bootstrapped over variants rather than compared as two independent numbers. At
these sample sizes two AUROCs can differ by less than the interval width of
either one.

**The pairing runs at two levels and they answer different questions.** Within a
subset, resampling variants asks whether this corpus is better served by one
side than the other. Across the 21 subsets, a signed-rank test on those 21
differences asks whether the side that wins does so consistently -- which is the
architecture-level question, and the one a single subset cannot answer.

Folds are never pooled. chr21 is 21% pathogenic, chr22 20%, chrX 48%, and AUROC
moves with class balance, so a value averaged over the three describes no
population.
"""
import argparse
import glob
import json
import os

import numpy as np

FOLDS = ["21", "22", "X"]


def auroc(labels, scores):
    """Rank-based, ties averaged -- the same estimator the fold analysis uses."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def signed_rank(diffs):
    """Wilcoxon signed-rank over the 21 subsets, normal approximation with ties.

    Zero differences are dropped, which is Wilcoxon's own convention: a subset
    that splits the two sides evenly carries no evidence either way.
    """
    d = np.asarray([x for x in diffs if x != 0], dtype=float)
    n = len(d)
    if n < 2:
        return {"n": n, "w": None, "z": None, "p_two_sided": None, "n_positive": 0}
    order = np.argsort(np.abs(d), kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    a = np.abs(d)[order]
    i, tie_term = 0, 0.0
    while i < n:
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
            t = j - i + 1
            tie_term += t ** 3 - t
        i = j + 1
    w_plus = float(ranks[d > 0].sum())
    mu = n * (n + 1) / 4.0
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0)
    z = (w_plus - mu) / sd if sd else 0.0
    from math import erfc
    return {"n": n, "w": w_plus, "z": float(z),
            "p_two_sided": float(erfc(abs(z) / (2 ** 0.5))),
            "n_positive": int((d > 0).sum())}


def read_runs(scores_dir):
    """{subset: {vcv_id: (chrom, label, score)}} from each run's predictions."""
    out = {}
    for path in sorted(glob.glob(os.path.join(scores_dir, "*", "predictions.jsonl"))):
        subset = os.path.basename(os.path.dirname(path))
        got = {}
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                r = json.loads(line)
                got[r["vcv_id"]] = (str(r.get("chrom", "")),
                                    r.get("label_pathogenic"), r.get("score"))
        if got:
            out[subset] = got
    return out


def compare(a_runs, b_runs, rounds, seed):
    """Per subset, per fold: AUROC(a) - AUROC(b) with a paired bootstrap CI."""
    rng = np.random.default_rng(seed)
    per_subset = {}
    for subset in sorted(set(a_runs) & set(b_runs)):
        a, b = a_runs[subset], b_runs[subset]
        shared = sorted(set(a) & set(b))
        folds = {}
        for fold in FOLDS:
            ids = [v for v in shared
                   if a[v][0] == fold and a[v][1] is not None
                   and a[v][2] is not None and b[v][2] is not None]
            if len(ids) < 4:
                continue
            labels = np.array([a[v][1] for v in ids])
            sa = np.array([a[v][2] for v in ids], dtype=float)
            sb = np.array([b[v][2] for v in ids], dtype=float)
            point = auroc(labels, sa) - auroc(labels, sb)
            # Resample the variants, not the runs: both sides scored these same
            # variants, so the pairing is preserved by drawing one index set and
            # applying it to both.
            draws = [auroc(labels[i], sa[i]) - auroc(labels[i], sb[i])
                     for i in (rng.integers(0, len(ids), len(ids))
                               for _ in range(rounds))]
            lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
            folds[fold] = {"n": len(ids), "diff": float(point), "ci": [lo, hi],
                           "favours_a": bool(lo > 0), "favours_b": bool(hi < 0)}
        if folds:
            per_subset[subset] = folds
    return per_subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", required=True, help="scores dir of side A")
    ap.add_argument("--b-dir", required=True, help="scores dir of side B")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1026)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a_runs, b_runs = read_runs(args.a_dir), read_runs(args.b_dir)
    shared = sorted(set(a_runs) & set(b_runs))
    print(f"  {args.a_name}: {len(a_runs)} 本   {args.b_name}: {len(b_runs)} 本   "
          f"共通 {len(shared)} 本", flush=True)
    only_a = sorted(set(a_runs) - set(b_runs))
    only_b = sorted(set(b_runs) - set(a_runs))
    if only_a or only_b:
        print(f"  片側にしか無い subset: {args.a_name}={only_a} {args.b_name}={only_b}")

    per_subset = compare(a_runs, b_runs, args.rounds, args.seed)

    out = {"a": args.a_name, "b": args.b_name, "rounds": args.rounds, "seed": args.seed,
           "pairing": "variants within a subset, then subsets within a fold",
           "per_subset": per_subset, "per_fold": {}}
    print(f"\n  {'fold':6s} {'n':>3s} {'差の中央':>9s} {'範囲':>19s} "
          f"{'A 有利':>7s} {'B 有利':>7s} {'符号検定 p':>11s}")
    for fold in FOLDS:
        diffs = [v[fold]["diff"] for v in per_subset.values() if fold in v]
        if not diffs:
            continue
        a_wins = sum(1 for v in per_subset.values()
                     if fold in v and v[fold]["favours_a"])
        b_wins = sum(1 for v in per_subset.values()
                     if fold in v and v[fold]["favours_b"])
        test = signed_rank(diffs)
        out["per_fold"][fold] = {"n_subsets": len(diffs),
                                 "median_diff": float(np.median(diffs)),
                                 "min_diff": float(min(diffs)), "max_diff": float(max(diffs)),
                                 "subsets_favouring_a": a_wins,
                                 "subsets_favouring_b": b_wins,
                                 "signed_rank": test}
        p = test["p_two_sided"]
        print(f"  chr{fold:4s} {len(diffs):>3d} {np.median(diffs):>+9.4f} "
              f"{min(diffs):>+9.4f}〜{max(diffs):<+9.4f} {a_wins:>7d} {b_wins:>7d} "
              f"{p:>11.2e}" if p is not None else "")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
