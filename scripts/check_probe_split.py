"""What the linear probe's folds actually are, read back from its own output.

The probe's table reports an ``overall`` AUROC and three per-chromosome ones,
and nothing in the table says which variants either was measured on. Three
questions keep coming back: whether the fit and the score share variants, what
the three chromosome columns are, and what ``overall`` covers. All three are
answerable from the files the campaign already wrote, so they are answered from
those rather than from the source.

Checks, per probe directory:

* the fold counts, and whether every run in the file agrees on them
* ``overall`` recomputed from ``predictions.jsonl`` against the recorded value.
  A pooled AUROC taken over the held-out scores of three separately fitted
  classifiers is not the same thing as an average of the three fold AUROCs, and
  this prints the gap so the size of the difference is on record rather than
  inferred.

The AUROC is the same rank statistic the probe uses, including its tie handling,
written out here so this check does not import what it is checking.
"""
import argparse
import json
import os


def auroc(labels, scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        mid = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    s = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (s - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fold_counts(entry):
    pf = entry["per_fold"]
    return tuple((f, pf[f]["n_train"], pf[f]["n_test"], pf[f]["pathogenic_test"])
                 for f in sorted(pf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", action="append", required=True,
                    help="a directory holding probe.json and <name>/predictions.jsonl")
    ap.add_argument("--recompute", default="",
                    help="comma-separated run names whose overall AUROC to "
                         "recompute from predictions.jsonl; empty for none")
    args = ap.parse_args()

    names = [n.strip() for n in args.recompute.split(",") if n.strip()]
    for d in args.probe_dir:
        got = json.load(open(os.path.join(d, "probe.json")))
        runs = got["runs"]
        print(f"\n=== {d} ===")
        print(f"  表の全体 {got['variants']:,} 変異   C={got['C']}  seed={got['seed']}")

        patterns = {fold_counts(v) for v in runs.values()}
        if "model_free" in got:
            patterns.add(fold_counts(got["model_free"]))
        print(f"  run {len(runs)} 本 + 対照   件数の組み合わせ {len(patterns)} 通り")
        for fold, n_train, n_test, n_path in sorted(patterns)[0]:
            print(f"    {fold}: 学習 {n_train:,} / 評価 {n_test:,}"
                  f"（病原性 {n_path:,}, {n_path / n_test:.1%}）")
        if len(patterns) > 1:
            print("    ** 件数が run ごとに違う。分割が揃っていない **")

        for name in names:
            if name not in runs:
                print(f"  {name}: この file に無い")
                continue
            path = os.path.join(d, name, "predictions.jsonl")
            if not os.path.exists(path):
                print(f"  {name}: predictions.jsonl が無い")
                continue
            rows = [json.loads(line) for line in open(path)]
            a = auroc([r["label_pathogenic"] for r in rows], [r["score"] for r in rows])
            stored = runs[name]["overall_auroc"]
            pf = runs[name]["per_fold"]
            total = sum(v["n_test"] for v in pf.values())
            weighted = sum(v["auroc"] * v["n_test"] for v in pf.values()) / total
            by_fold = {}
            for r in rows:
                by_fold[r["fold"]] = by_fold.get(r["fold"], 0) + 1
            print(f"  {name}: 保留側 {len(rows):,} 件 {by_fold}")
            print(f"    overall 再計算 {a:.6f}  記録 {stored:.6f}  差 {abs(a - stored):.2e}")
            print(f"    overall − 件数加重の fold 平均 {stored - weighted:+.4f}"
                  f"（fold ごとに別の分類器なので一致しない）")


if __name__ == "__main__":
    main()
