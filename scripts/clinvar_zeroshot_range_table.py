"""AUROC of the 21 bert-w1026 runs, one row per run and summing range.

§2 of the instruction of 2026-11-02: the difference between the reference and
the variant sequence is not collapsed into one number. Every position in the
window is masked and scored anyway, so narrowing the range only changes which
of the already-computed values are added, and the four ranges cost one pass:

  window     all 1,024 positions   the pseudo-log-likelihood of the window
  flank128   257 positions         the variant and 128 bases either side
  flank32    65 positions
  point      1 position            the variant position alone

The ranges are means rather than sums. AUROC is invariant to dividing every
score by the same constant, so this changes no value in the table; it keeps the
columns on one scale for anybody reading the per-variant files.

The three chromosomes are kept apart and summarised by a mean weighted by how
many variants each carries, because chrX is about 48 per cent pathogenic
against about 20 per cent for chr21 and chr22 and AUROC moves with the class
balance. chrY's 29 variants are in the per-variant files and in no column
here: they are never a fold.
"""
import argparse
import csv
import glob
import json
import os

FOLDS = ("21", "22", "X")
RANGES = ("window", "flank128", "flank32", "point")


def group_of(subset):
    for g in ("global_random", "eukaryote_matched_random", "mammal_centered"):
        if subset.startswith(g):
            return g
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True,
                    help="directory of window-<subset>.json from "
                         "clinvar_zeroshot_bert_rescore.py")
    ap.add_argument("--out", required=True, help="TSV to write")
    ap.add_argument("--expect-runs", type=int, default=21)
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.scores_dir, "window-*.json"))):
        d = json.load(open(path))
        subset = os.path.basename(path)[len("window-"):-len(".json")]
        if "ranges" not in d:
            raise SystemExit(f"{path}: no 'ranges' block -- scored before the "
                             f"ranges existed, so it cannot go in this table")
        for r in RANGES:
            b = d["ranges"][r]
            num = sum(b[f"n_chr{c}"] * b[f"auroc_chr{c}"] for c in FOLDS)
            den = sum(b[f"n_chr{c}"] for c in FOLDS)
            rows.append({
                "subset": subset, "group": group_of(subset), "range": r,
                "n_positions": b["n_positions"],
                "checkpoint_step": d.get("checkpoint_step", ""),
                "n_chr21": b["n_chr21"], "n_chr22": b["n_chr22"],
                "n_chrX": b["n_chrX"], "n_three": den,
                "auroc_chr21": round(b["auroc_chr21"], 6),
                "auroc_chr22": round(b["auroc_chr22"], 6),
                "auroc_chrX": round(b["auroc_chrX"], 6),
                "auroc_weighted_mean": round(num / den, 6),
            })

    n_runs = len({r["subset"] for r in rows})
    if n_runs != args.expect_runs:
        raise SystemExit(f"{n_runs} runs in {args.scores_dir}, expected "
                         f"{args.expect_runs}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = list(rows[0])
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {args.out}  ({len(rows)} 行 = {n_runs} run x "
          f"{len(RANGES)} 範囲)")

    print("\n  === 範囲ごと: 件数加重平均の分布（21 run） ===")
    for r in RANGES:
        v = sorted(x["auroc_weighted_mean"] for x in rows if x["range"] == r)
        mid = v[len(v) // 2] if len(v) % 2 else (v[len(v)//2 - 1] + v[len(v)//2]) / 2
        n_pos = next(x["n_positions"] for x in rows if x["range"] == r)
        print(f"  {r:9s} ({n_pos:>5,} 箇所)  最小 {v[0]:.4f}  中央 {mid:.4f}  "
              f"最大 {v[-1]:.4f}")

    print("\n  === 範囲を変えると run ごとにどう動くか（window を基準に） ===")
    by = {}
    for x in rows:
        by.setdefault(x["subset"], {})[x["range"]] = x["auroc_weighted_mean"]
    print(f"  {'subset':36s} " + "  ".join(f"{r:>10s}" for r in RANGES))
    for s in sorted(by, key=lambda k: -by[k]["window"]):
        print(f"  {s:36s} " + "  ".join(f"{by[s][r]:10.4f}" for r in RANGES))
    better = {r: sum(1 for s in by if by[s][r] > by[s]["window"]) for r in RANGES}
    print("\n  window より高い run の数: " +
          "  ".join(f"{r} {better[r]}/{len(by)}" for r in RANGES if r != "window"))


if __name__ == "__main__":
    main()
