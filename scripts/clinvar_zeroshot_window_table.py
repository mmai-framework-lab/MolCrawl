"""One table of ClinVar zero-shot AUROCs, one row per measurement.

The three chromosomes are reported separately and summarised by a mean weighted
by how many variants each contributes. The value that pools all three into one
ranking is deliberately absent: the three carry different base rates -- chrX is
about 48 per cent pathogenic against about 20 per cent for chr21 and chr22 --
so a pooled ranking is not the same quantity as the per-chromosome ones, and
quoting both invites the two being read as interchangeable.

Zero-shot fits nothing, so no chromosome is held out from any training. The
chromosome column names the rows a value was computed over, and chrY, which is
29 variants and never a test set, is left out of all of them.

Two kinds of source are read, because the campaign's series are stored
differently:

  a directory of ``window-<subset>.json`` files, which carry the per-chromosome
      AUROCs the rescorer computed as it went
  a directory of ``<subset>/predictions.jsonl`` files, one row per variant, from
      which the per-chromosome AUROCs are computed here

Both paths end in the same columns, so the series can be compared row for row.

The ``input_form`` column is not a formatting choice. ``wrapped`` is
``[CLS]`` + 1,024 bases + ``[SEP]`` = 1,026 tokens, which is what the genome
BERT runs were trained on; ``bare`` is the 1,024 bases alone, which is what
nanoGPT was trained on. The two architectures therefore read as "not matched"
in the column while each is in fact being scored in its own training-time
form, so the table carries one note line saying so.
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


def rows_from_window_json(path):
    """Per-chromosome (n, auroc) as the rescorer already recorded them."""
    got = json.load(open(path))
    per = {}
    for fold in FOLDS:
        n, value = got.get(f"n_chr{fold}"), got.get(f"auroc_chr{fold}")
        if n and value is not None:
            per[fold] = (int(n), float(value))
    return got.get("input_form"), got.get("positions"), per, int(got.get("n_variants", 0))


def full_scorings(paths):
    """The scorings that covered the whole variant table, and why the rest went.

    The same directory holds timing probes, which score a prefix of the table
    with --max-variants. They are the same file shape as a full scoring and
    would otherwise be averaged in beside it, so anything short of the largest
    table seen is dropped and named.
    """
    read = {}
    for path in paths:
        subset = os.path.basename(path)[len("window-"):-len(".json")]
        read[path] = (subset,) + rows_from_window_json(path)
    if not read:
        return [], []
    full = max(rec[4] for rec in read.values())
    keep = [(p, rec) for p, rec in read.items() if rec[4] == full]
    drop = [(p, rec) for p, rec in read.items() if rec[4] != full]
    return sorted(keep), sorted(drop)


def rows_from_predictions(path):
    """Per-chromosome (n, auroc) computed from the per-variant scores."""
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
    return per


def weighted_mean(per):
    total = sum(n for n, _a in per.values())
    if not total:
        return None, 0
    return sum(n * a for n, a in per.values()) / total, total


def emit(out, series, subset, input_form, positions, per):
    for fold in FOLDS:
        if fold not in per:
            continue
        n, value = per[fold]
        out.append({"series": series, "subset": subset, "input_form": input_form,
                    "positions": positions, "chrom": f"chr{fold}",
                    "n_variants": n, "auroc": value})
    mean, total = weighted_mean(per)
    if mean is not None:
        out.append({"series": series, "subset": subset, "input_form": input_form,
                    "positions": positions, "chrom": "weighted_mean",
                    "n_variants": total, "auroc": mean})


def note_line(rows, extra):
    """One comment line: what each series' input_form is, and what it means.

    Read on its own, the column looks like two series that were not matched.
    The forms per series come from the rows, so the line cannot drift from the
    table it sits on; anything the rows do not carry comes in through --note.
    """
    forms = dict.fromkeys((r["series"], r["input_form"]) for r in rows)
    listed = "、".join(f"{series}={form}" for series, form in forms)
    parts = ["input_form は採点時に入力した形である。"
             "wrapped は [CLS]+1,024+[SEP] の 1,026 トークンで、genome BERT の学習時の形。"
             "bare は特別なトークンを持たない 1,024 トークンで、nanoGPT の学習時の形。"
             f"この表の系列: {listed}。"]
    parts.extend(extra)
    return "# " + "".join(parts) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-dir", action="append", default=[],
                    metavar="series:path",
                    help="repeatable: a directory of window-<subset>.json files")
    ap.add_argument("--predictions-dir", action="append", default=[],
                    metavar="series:input_form:positions:path",
                    help="repeatable: a directory of <subset>/predictions.jsonl")
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--note", action="append", default=[],
                    help="appended to the note line, for what the rows cannot say")
    args = ap.parse_args()

    out = []
    for spec in args.window_dir:
        series, root = spec.split(":", 1)
        keep, drop = full_scorings(glob.glob(os.path.join(root, "window-*.json")))
        for _path, (subset, input_form, positions, per, _n) in keep:
            emit(out, series, subset, input_form, positions, per)
        print(f"  {series}: {len(keep)} 本  ({root})")
        for path, (subset, _f, _p, _per, n) in drop:
            print(f"    除外 {subset}: {n:,} 件しか採点していない  "
                  f"({os.path.basename(path)})")

    for spec in args.predictions_dir:
        series, input_form, positions, root = spec.split(":", 3)
        found = sorted(glob.glob(os.path.join(root, "*", "predictions.jsonl")))
        for path in found:
            subset = os.path.basename(os.path.dirname(path))
            emit(out, series, subset, input_form, positions,
                 rows_from_predictions(path))
        print(f"  {series}: {len(found)} 本  ({root})")

    if not out:
        raise SystemExit("no measurements found -- check the directories given")

    fields = ["series", "subset", "input_form", "positions", "chrom",
              "n_variants", "auroc"]
    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    with open(args.out_tsv, "w", newline="") as handle:
        handle.write(note_line(out, args.note))
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for rec in out:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in rec.items()})

    print()
    for series in dict.fromkeys(r["series"] for r in out):
        means = sorted(r["auroc"] for r in out
                       if r["series"] == series and r["chrom"] == "weighted_mean")
        if means:
            mid = means[len(means) // 2] if len(means) % 2 else \
                (means[len(means) // 2 - 1] + means[len(means) // 2]) / 2
            print(f"  {series:12s} 加重平均 {len(means)} 本: "
                  f"中央 {mid:.4f}  最小 {means[0]:.4f}  最大 {means[-1]:.4f}")
    print(f"\n  wrote {args.out_tsv}  ({len(out):,} 行)")


if __name__ == "__main__":
    main()
