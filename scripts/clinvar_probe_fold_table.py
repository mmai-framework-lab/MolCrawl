"""One table of the linear probe's per-chromosome AUROCs, one row per measurement.

The probe's chromosomes are not the zero-shot table's chromosomes. Zero-shot
fits nothing, so there its chromosome column only names the rows a value was
computed over. The probe fits a logistic regression per fold on every variant
outside the held-out chromosome and scores the held-out chromosome alone, so
here the chromosome *is* the train/test split and three separate classifiers
are involved. The count of variants each classifier was fitted on is therefore
carried beside the count it was scored on; a reader cannot tell the two tables
apart on the AUROC column alone.

The pooled ``overall`` is deliberately absent, as it is from the zero-shot
table, and for a second reason on top of the differing base rates: it ranks the
probabilities of three separately fitted classifiers against each other, and
their scales are not tied to one another.

Values are read from each series' ``probe.json``, where the campaign wrote
them, rather than recomputed from ``predictions.jsonl``, so the rows match the
numbers already reported. ``--verify-predictions`` recomputes one series from
its per-variant scores and reports the largest disagreement.

The model-free control uses no model, so it exists once rather than once per
subset: its block is identical in every ``probe.json`` written by the campaign,
and the subset column says ``-`` on its rows.
"""
import argparse
import csv
import glob
import json
import os

FOLDS = ("chr21", "chr22", "chrX")


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


def weighted_mean(per):
    """The folds' AUROCs averaged by how many variants each was scored on."""
    num = sum(rec["auroc"] * rec["n_test"] for rec in per.values())
    den = sum(rec["n_test"] for rec in per.values())
    return (num / den, den) if den else (None, 0)


def emit(out, series, subset, per):
    for fold in FOLDS:
        rec = per.get(fold)
        if rec is None:
            continue
        out.append({"series": series, "subset": subset, "chrom": fold,
                    "n_train": rec["n_train"], "n_variants": rec["n_test"],
                    "auroc": rec["auroc"]})
    mean, total = weighted_mean({k: v for k, v in per.items() if k in FOLDS})
    if mean is not None:
        out.append({"series": series, "subset": subset, "chrom": "weighted_mean",
                    "n_train": "-", "n_variants": total, "auroc": mean})


def per_fold(block):
    """{fold: {auroc, n_test, n_train}} from a probe.json run or control block."""
    got = {}
    for fold, rec in (block.get("per_fold") or {}).items():
        if rec.get("auroc") is None:
            continue
        got[fold] = {"auroc": rec["auroc"], "n_test": rec["n_test"],
                     "n_train": rec["n_train"]}
    return got


def verify(root, runs):
    """Recompute one series from predictions.jsonl; return the largest gap."""
    worst = (None, None, 0.0)
    for subset, block in sorted(runs.items()):
        path = os.path.join(root, subset, "predictions.jsonl")
        if not os.path.exists(path):
            continue
        rows = {}
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("label_pathogenic") is None or r.get("score") is None:
                    continue
                rows.setdefault(r.get("fold"), []).append(
                    (r["label_pathogenic"], r["score"]))
        for fold, got in rows.items():
            said = (block.get("per_fold") or {}).get(fold, {}).get("auroc")
            if said is None:
                continue
            mine = auroc([y for y, _ in got], [s for _, s in got])
            gap = abs(mine - said)
            if gap > worst[2]:
                worst = (subset, fold, gap)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True,
                    metavar="name:path",
                    help="repeatable: a series name and its probe directory")
    ap.add_argument("--control-from", default="",
                    help="the probe directory whose model_free block to emit")
    ap.add_argument("--control-name", default="model_free")
    ap.add_argument("--expect-subsets", type=int, default=21,
                    help="refuse a series with a different count (0 disables)")
    ap.add_argument("--verify-predictions", default="",
                    metavar="name:path",
                    help="recompute this series from its per-variant scores")
    ap.add_argument("--out-tsv", required=True)
    args = ap.parse_args()

    out = []
    for spec in args.series:
        name, root = spec.split(":", 1)
        runs = json.load(open(os.path.join(root, "probe.json")))["runs"]
        if args.expect_subsets and len(runs) != args.expect_subsets:
            raise SystemExit(f"REFUSING: {name} has {len(runs)} subsets, "
                             f"not {args.expect_subsets} ({root})")
        for subset, block in sorted(runs.items()):
            emit(out, name, subset, per_fold(block))
        print(f"  {name}: {len(runs)} 本  ({root})")

    if args.control_from:
        block = json.load(open(os.path.join(args.control_from, "probe.json")))
        control = block.get("model_free")
        if not control:
            raise SystemExit(f"no model_free block in {args.control_from}")
        emit(out, args.control_name, "-", per_fold(control))
        print(f"  {args.control_name}: 1 組  ({args.control_from})")
        # The control fits no model, so every probe.json should carry the same
        # block. Say so out loud rather than leaving the reader to wonder which
        # directory this one came from.
        seen = {}
        for path in sorted(glob.glob(os.path.join(
                os.path.dirname(args.control_from.rstrip("/")),
                "clinvar-probe-*", "probe.json"))):
            mf = (json.load(open(path)) or {}).get("model_free")
            if mf:
                key = tuple(round(v["auroc"], 12)
                            for v in (mf.get("per_fold") or {}).values())
                seen.setdefault(key, []).append(os.path.basename(
                    os.path.dirname(path)))
        print(f"    model_free を持つ probe.json {sum(len(v) for v in seen.values())} 本、"
              f"値の組は {len(seen)} 通り")

    if args.verify_predictions:
        name, root = args.verify_predictions.split(":", 1)
        runs = json.load(open(os.path.join(root, "probe.json")))["runs"]
        subset, fold, gap = verify(root, runs)
        print(f"  照合 {name}: predictions.jsonl から再計算した最大差 "
              f"{gap:.2e}" + (f"（{subset} {fold}）" if subset else "（照合なし）"))

    fields = ["series", "subset", "chrom", "n_train", "n_variants", "auroc"]
    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    with open(args.out_tsv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for rec in out:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in rec.items()})

    print()
    for series in dict.fromkeys(r["series"] for r in out):
        means = sorted(r["auroc"] for r in out
                       if r["series"] == series and r["chrom"] == "weighted_mean")
        mid = means[len(means) // 2] if len(means) % 2 else \
            (means[len(means) // 2 - 1] + means[len(means) // 2]) / 2
        print(f"  {series:18s} 加重平均 {len(means):2d} 本: 中央 {mid:.4f}  "
              f"最小 {means[0]:.4f}  最大 {means[-1]:.4f}")
    print(f"\n  wrote {args.out_tsv}  ({len(out):,} 行)")


if __name__ == "__main__":
    main()
