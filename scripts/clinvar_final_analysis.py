"""Fold-wise ClinVar analysis: did pretraining beat the shared statistics?

Three things the combined number cannot say.

**Folds are not interchangeable.** chr21 is 21% pathogenic, chr22 20%, chrX 48%,
and AUROC moves with class balance. A single value over the union mixes three
different compositions, so every figure here is also given per fold.

**Beating a baseline is a paired question.** Both the model and the Markov model
score the same variants, so the difference can be taken variant by variant and
bootstrapped over variants. A model is said to beat the baseline when that
interval excludes zero -- not when its point estimate is higher, which at these
sample sizes it can be by less than the interval width.

**A group difference needs a group test.** Whether `global_random` sits above
`eukaryote_matched` is a question about two sets of ten independent subsets, so
it gets a rank test rather than a comparison of the two ranges.

The correlations are there to check whether the group difference is explained by
something other than corpus composition: how much of each subset was lost to
windowing, and how far each run got from its own degenerate baseline. The second
uses the margin rather than raw loss because the 21 valid splits are not the same
data and their losses are not on one scale.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

FOLDS = ["21", "22", "X"]        # chrY is 29 variants, never a fold


def auroc(labels, scores):
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


def paired_ci(labels, a, b, idx):
    """Bootstrap the difference of two AUROCs over the variants they share."""
    d = np.array([auroc(labels[i], a[i]) - auroc(labels[i], b[i]) for i in idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def mann_whitney(x, y):
    """Rank-sum on two small independent groups; normal approximation with ties."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    allv = np.concatenate([x, y])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=float)
    s = allv[order]
    i = 0
    tie_term = 0.0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
            t = j - i + 1
            tie_term += t ** 3 - t
        i = j + 1
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    sd = math_sqrt((n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1))))
    z = (u1 - mu) / sd if sd else 0.0
    from math import erfc
    p = erfc(abs(z) / (2 ** 0.5))            # two-sided
    # AUC-style effect size: probability a random x exceeds a random y
    return {"u": float(u1), "z": float(z), "p_two_sided": float(p),
            "prob_x_gt_y": float(u1 / (n1 * n2))}


def math_sqrt(v):
    return v ** 0.5 if v > 0 else 0.0


def pearson_spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        r[o] = np.arange(1, len(v) + 1, dtype=float)
        return r
    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        d = (a.std() * b.std())
        return float((a * b).mean() / d) if d else float("nan")
    return corr(x, y), corr(rank(x), rank(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--baseline-scores", required=True,
                    help="per-variant JSONL from clinvar_zeroshot_baselines.py")
    ap.add_argument("--window-loss", default="")
    ap.add_argument("--degenerate-baselines", default="")
    ap.add_argument("--per-run-csv", default="", help="b21-per-run.csv, for best_val")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1026)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base = {json.loads(x)["vcv_id"]: json.loads(x)
            for x in open(args.baseline_scores)}
    order = sorted(base)
    labels = np.array([base[v]["label_pathogenic"] for v in order])
    chrom = np.array([base[v]["chrom"] for v in order])
    mk = np.array([base[v]["markov"] for v in order], dtype=float)

    runs = {}
    for f in sorted(glob.glob(os.path.join(args.scores_dir, "*", "predictions.jsonl"))):
        s = os.path.basename(os.path.dirname(f))
        rec = {json.loads(x)["vcv_id"]: json.loads(x)["score"] for x in open(f)}
        if set(rec) != set(order):
            raise SystemExit(f"{s}: scores a different variant set from the baseline")
        runs[s] = np.array([rec[v] for v in order], dtype=float)

    print(f"  runs {len(runs)}   variants {len(order):,}")
    print(f"  markov baseline overall {auroc(labels, mk):.4f}")
    print("\n  === fold ごとの構成 ===")
    for c in FOLDS:
        sel = chrom == c
        print(f"  chr{c:<3s} {int(sel.sum()):>6,} 件   病原性 {labels[sel].mean():.1%}   "
              f"markov {auroc(labels[sel], mk[sel]):.4f}")

    rng = np.random.default_rng(args.seed)
    fold_idx = {c: rng.integers(0, int((chrom == c).sum()),
                                size=(args.rounds, int((chrom == c).sum())))
                for c in FOLDS}

    out = {"markov_overall": auroc(labels, mk), "folds": {}, "runs": {}}
    for c in FOLDS:
        sel = chrom == c
        out["folds"][c] = {"n": int(sel.sum()), "pathogenic_rate": float(labels[sel].mean()),
                           "markov_auroc": auroc(labels[sel], mk[sel])}

    print("\n  === 各 run: fold 別 AUROC と、5-mer マルコフとの対応のある差 ===")
    print(f"  {'subset':36s} " + "  ".join(f"{'chr'+c:>21s}" for c in FOLDS))
    for s in sorted(runs, key=lambda k: -auroc(labels, runs[k])):
        v = runs[s]
        row, line = {}, []
        for c in FOLDS:
            sel = chrom == c
            a = auroc(labels[sel], v[sel])
            lo, hi = paired_ci(labels[sel], v[sel], mk[sel], fold_idx[c])
            beat = lo > 0
            row[c] = {"auroc": a, "diff_vs_markov": a - auroc(labels[sel], mk[sel]),
                      "diff_ci": [lo, hi], "beats_baseline": bool(beat)}
            line.append(f"{a:.4f} {'+' if beat else ' '}[{lo:+.3f},{hi:+.3f}]")
        n_beat = sum(1 for c in FOLDS if row[c]["beats_baseline"])
        out["runs"][s] = {"per_fold": row, "folds_beaten": n_beat,
                          "overall_auroc": auroc(labels, v)}
        print(f"  {s:36s} " + "  ".join(line) + f"   {n_beat}/3")

    won = [s for s, t in out["runs"].items() if t["folds_beaten"] == len(FOLDS)]
    print(f"\n  3 fold すべてで基準線を越えた subset: {len(won)} / {len(runs)}")
    for s in sorted(won, key=lambda k: -out["runs"][k]["overall_auroc"]):
        print(f"    {s}")

    # --- group test -------------------------------------------------------
    g1 = [t["overall_auroc"] for s, t in out["runs"].items() if s.startswith("global_random")]
    g2 = [t["overall_auroc"] for s, t in out["runs"].items()
          if s.startswith("eukaryote_matched")]
    if g1 and g2:
        mw = mann_whitney(g1, g2)
        out["group_test"] = {"global_random_n": len(g1), "eukaryote_matched_n": len(g2),
                             **mw}
        print("\n  === 群間（mammal_centered は 1 本なので除外） ===")
        print(f"  global_random n={len(g1)} 中央 {np.median(g1):.4f}   "
              f"eukaryote_matched n={len(g2)} 中央 {np.median(g2):.4f}")
        print(f"  Mann-Whitney U={mw['u']:.0f}  z={mw['z']:.3f}  p={mw['p_two_sided']:.2e}"
              f"  P(global>eukaryote)={mw['prob_x_gt_y']:.3f}")

    # --- correlations -----------------------------------------------------
    xs = {}
    if args.window_loss and os.path.exists(args.window_loss):
        xs["窓の欠落率"] = {r["subset"]: r["loss_fraction"]
                        for r in json.load(open(args.window_loss))}
    if (args.degenerate_baselines and args.per_run_csv
            and os.path.exists(args.degenerate_baselines) and os.path.exists(args.per_run_csv)):
        deg = {r["subset"]: r["gpt2"]["baseline"]
               for r in json.load(open(args.degenerate_baselines))}
        best = {r["subset"]: float(r["best_val"])
                for r in csv.DictReader(open(args.per_run_csv))}
        xs["退化解からの余裕"] = {s: deg[s] - best[s] for s in deg if s in best}
    if xs:
        print("\n  === 相関（21 subset） ===")
        out["correlations"] = {}
        for name, m in xs.items():
            common = [s for s in out["runs"] if s in m]
            x = [m[s] for s in common]
            y = [out["runs"][s]["overall_auroc"] for s in common]
            p, sp = pearson_spearman(x, y)
            out["correlations"][name] = {"n": len(common), "pearson": p, "spearman": sp}
            print(f"  {name:16s} n={len(common)}  Pearson {p:+.3f}  Spearman {sp:+.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
