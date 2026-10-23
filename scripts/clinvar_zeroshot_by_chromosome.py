"""Zero-shot AUROC per chromosome, and an overall that matches the folds.

The overall value the evaluator writes covers every row it scored, and the
scored table is chr21, chr22, chrX and chrY. The fold-wise analysis uses three
of those: chrY is 29 variants and is not a test set. So the overall and the
per-fold numbers describe different populations, and the difference was never
written down.

Both are produced here. The chrY-inclusive value is kept because it is what
every earlier report quoted, and the counts go beside each so the two can be
told apart on sight.

Read from each subset's predictions.jsonl, which carries one row per variant
with its chromosome, its label and its score -- the same rows the evaluator
computed its own AUROC from.
"""
import argparse
import csv
import glob
import json
import os

FOLDS = ("21", "22", "X")


def auroc(labels, scores):
    """Rank-based AUROC with ties averaged. No numpy: this runs on lists."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        shared = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[k] = shared
        i = j + 1
    n_pos = sum(1 for _s, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum = sum(r for r, (_s, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def read_predictions(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append((str(r.get("chrom", "")).replace("chr", ""),
                         r.get("label_pathogenic"), r.get("score")))
    return [r for r in rows if r[1] is not None and r[2] is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", action="append", required=True,
                    metavar="series:path",
                    help="repeatable: one zero-shot directory per series")
    ap.add_argument("--out-tsv", required=True)
    args = ap.parse_args()

    out = []
    for spec in args.scores_dir:
        series, root = spec.split(":", 1)
        for path in sorted(glob.glob(os.path.join(root, "*", "predictions.jsonl"))):
            subset = os.path.basename(os.path.dirname(path))
            rows = read_predictions(path)
            whole = [(y, s) for _c, y, s in rows]
            three = [(y, s) for c, y, s in rows if c in FOLDS]
            rec = {"series": series, "subset": subset,
                   "n_all": len(whole), "auroc_all": auroc([y for y, _ in whole],
                                                           [s for _, s in whole]),
                   "n_three": len(three), "auroc_three": auroc([y for y, _ in three],
                                                               [s for _, s in three])}
            for fold in FOLDS:
                got = [(y, s) for c, y, s in rows if c == fold]
                rec[f"n_chr{fold}"] = len(got)
                rec[f"auroc_chr{fold}"] = auroc([y for y, _ in got], [s for _, s in got])
            out.append(rec)
            print(f"  {series:12s} {subset:34s} "
                  f"全 {rec['n_all']:,} {rec['auroc_all']:.4f}  "
                  f"3 本 {rec['n_three']:,} {rec['auroc_three']:.4f}  "
                  f"差 {rec['auroc_three'] - rec['auroc_all']:+.4f}")

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    with open(args.out_tsv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(out[0]))
        writer.writeheader()
        for rec in out:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in rec.items()})
    print(f"\n  wrote {args.out_tsv}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
