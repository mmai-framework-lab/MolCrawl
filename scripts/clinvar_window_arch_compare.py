"""The architecture comparison, taken at one scoring formula on both sides.

The comparison in the §7 table of the 2026-10-15 report puts BERT's zero-shot
AUROC beside GPT-2's, but the two were not the same quantity: BERT masked the
variant position alone and GPT-2 took the log-likelihood of the whole 1,024
token window. Read as a table the rows look like one measurement each, and
nothing in them says the formula differed.

So two pairings come out here, side by side:

  window   BERT over the whole window against GPT-2 over the whole window --
           the same formula on both sides, which is what the 21 window
           scorings were produced for
  variant  BERT at the variant position against GPT-2 over the whole window --
           the pairing the earlier table reported, kept so the change from
           fixing the formula can be read off rather than inferred

Differences are paired by subset and signed GPT-2 minus BERT, so a negative
value means BERT scored higher. That is the sign convention the §7 table
already uses.

Per fold, and a mean weighted by how many variants each fold holds. The value
that pools the three folds into one ranking is not here: the folds carry
different base rates, so the pooled ranking is a different quantity.

The signed-rank test is Wilcoxon's, with the normal approximation and no tie
correction, which is what produced the figures in the earlier table (n=21, all
one sign gives w=0, z=-4.015, p=6.0e-5).
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics

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


def signed_rank(diffs):
    """Wilcoxon signed-rank, normal approximation, zeros dropped.

    No tie correction, because the table this feeds was produced that way and a
    corrected z would not be comparable to the rows already reported.
    """
    kept = [d for d in diffs if d != 0]
    n = len(kept)
    if n < 2:
        return None
    order = sorted(range(n), key=lambda i: abs(kept[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(kept[order[j + 1]]) == abs(kept[order[i]]):
            j += 1
        shared = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    w_pos = sum(r for r, d in zip(ranks, kept) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, kept) if d < 0)
    w = min(w_pos, w_neg)
    mean = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w - mean) / sd
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"n": n, "w": w, "z": z, "p": p}


def per_fold_from_window_json(root):
    """{subset: {fold: (n, auroc)}} from the window rescorer's own output.

    The timing probes live in the same directory with the same file shape and
    score a prefix of the table, so anything short of the largest table seen is
    dropped -- the same rule the reporting table uses.
    """
    read = {}
    for path in glob.glob(os.path.join(root, "window-*.json")):
        got = json.load(open(path))
        subset = os.path.basename(path)[len("window-"):-len(".json")]
        per = {}
        for fold in FOLDS:
            n, value = got.get(f"n_chr{fold}"), got.get(f"auroc_chr{fold}")
            if n and value is not None:
                per[fold] = (int(n), float(value))
        read[subset] = (per, int(got.get("n_variants", 0)))
    if not read:
        return {}
    full = max(n for _per, n in read.values())
    return {s: per for s, (per, n) in read.items() if n == full}


def per_fold_from_predictions(root):
    """{subset: {fold: (n, auroc)}} computed from the per-variant scores."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "predictions.jsonl"))):
        subset = os.path.basename(os.path.dirname(path))
        by_fold = {fold: ([], []) for fold in FOLDS}
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                got = json.loads(line)
                chrom = str(got.get("chrom", "")).replace("chr", "")
                label, score = got.get("label_pathogenic"), got.get("score")
                if chrom not in by_fold or label is None or score is None:
                    continue
                by_fold[chrom][0].append(label)
                by_fold[chrom][1].append(score)
        per = {}
        for fold, (labels, scores) in by_fold.items():
            if labels:
                value = auroc(labels, scores)
                if value is not None:
                    per[fold] = (len(labels), value)
        out[subset] = per
    return out


def weighted(per):
    total = sum(n for n, _a in per.values())
    if not total:
        return None
    return sum(n * a for n, a in per.values()) / total


def value_at(per, key):
    if key == "weighted_mean":
        return weighted(per)
    got = per.get(key.replace("chr", ""))
    return got[1] if got else None


def compare(name, left, right, keys):
    """One block of rows: GPT-2 minus BERT, paired by subset, per key."""
    shared = sorted(set(left) & set(right))
    rows = []
    paired = []
    for key in keys:
        diffs = []
        for subset in shared:
            a, b = value_at(left[subset], key), value_at(right[subset], key)
            if a is None or b is None:
                continue
            diffs.append(b - a)
            paired.append({"pairing": name, "fold": key, "subset": subset,
                           "bert": a, "gpt2": b, "diff": b - a})
        if not diffs:
            continue
        test = signed_rank(diffs) or {}
        rows.append({"pairing": name, "fold": key, "n_subsets": len(diffs),
                     "median": statistics.median(diffs),
                     "low": min(diffs), "high": max(diffs),
                     "gpt2_higher": sum(1 for d in diffs if d > 0),
                     "bert_higher": sum(1 for d in diffs if d < 0),
                     "w": test.get("w"), "z": test.get("z"), "p": test.get("p")})
    return rows, paired


def write_tsv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        for rec in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in rec.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-dir", required=True,
                    help="BERT whole-window scorings (window-<subset>.json)")
    ap.add_argument("--bert-variant-dir", required=True,
                    help="BERT variant-position scores (<subset>/predictions.jsonl)")
    ap.add_argument("--gpt2-dir", required=True,
                    help="GPT-2 whole-window scores (<subset>/predictions.jsonl)")
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--out-paired-tsv", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--expect-subsets", type=int, default=21)
    args = ap.parse_args()

    bert_window = per_fold_from_window_json(args.window_dir)
    bert_variant = per_fold_from_predictions(args.bert_variant_dir)
    gpt2 = per_fold_from_predictions(args.gpt2_dir)
    print(f"  BERT 窓全体   {len(bert_window)} 本")
    print(f"  BERT 変異 1 点 {len(bert_variant)} 本")
    print(f"  GPT-2 窓全体  {len(gpt2)} 本")
    if len(bert_window) != args.expect_subsets:
        raise SystemExit(f"REFUSING: {len(bert_window)} window scorings, "
                         f"expected {args.expect_subsets}")

    keys = ["chr21", "chr22", "chrX", "weighted_mean"]
    rows, paired = [], []
    for name, left in (("window", bert_window), ("variant", bert_variant)):
        got_rows, got_paired = compare(name, left, gpt2, keys)
        rows += got_rows
        paired += got_paired

    write_tsv(args.out_tsv, rows)
    print(f"\n  wrote {args.out_tsv}  ({len(rows)} 行)")
    if args.out_paired_tsv:
        write_tsv(args.out_paired_tsv, paired)
        print(f"  wrote {args.out_paired_tsv}  ({len(paired)} 行)")

    # The difference is gpt2 - bert, so the label has to read that way round:
    # a negative median means BERT scored higher.
    label = {"window": "GPT-2 窓全体 − BERT 窓全体（式を揃えた組）",
             "variant": "GPT-2 窓全体 − BERT 変異 1 点（従来の組・式が違う）"}
    lines = ["| 組 | 区分 | 本数 | 差の中央値 | 範囲 | GPT-2 有利 / BERT 有利 | 符号付き順位 |",
             "|---|---|---:|---:|---|---|---|"]
    for rec in rows:
        test = ("—" if rec["w"] is None else
                f"w={rec['w']:.0f}、z={rec['z']:.3f}、p={rec['p']:.1e}")
        lines.append(
            f"| {label[rec['pairing']]} | {rec['fold']} | {rec['n_subsets']} | "
            f"{rec['median']:+.4f} | {rec['low']:+.4f}〜{rec['high']:+.4f} | "
            f"{rec['gpt2_higher']} / {rec['bert_higher']} | {test} |")
    table = "\n".join(lines)
    print()
    print(table)
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w") as handle:
            handle.write(table + "\n")
        print(f"\n  wrote {args.out_md}")


if __name__ == "__main__":
    main()
