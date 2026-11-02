"""Hold one scoring pass against an earlier one, value by value.

The 21 bert-w1026 runs have now been scored over the whole window three times:
once for the aggregates alone, once (2026-10-30) writing rows that carry only
``{chrom, fold, label_pathogenic, score}``, and once (2026-11-02) writing the
identifiers and the four summing ranges. The reported fold table is built from
the first of those, so the question this answers is whether the new pass
reproduces it or quietly replaces it.

Two comparisons, because agreement can fail in two different places:

  the aggregates  AUROC per chromosome, read out of the ``window-<subset>.json``
      files. A difference here moves a reported number.
  the per-variant values  the new ``score_window`` against the old ``score``.
      The whole-window value is still accumulated as one reduction per chunk on
      the device added in chunk order, so this is expected to be identical bit
      for bit, not merely close -- and if it is not, the ranges were implemented
      at the cost of the number they were added beside.

The rows of 2026-10-30 carry no ``vcv_id``, which is why they are being
re-scored, so they cannot be joined on the variant. Both passes walk the same
variant table in order, so they are joined on the line number instead, and the
chromosome and the label on each line are required to agree -- that is as much
of a check on the alignment as rows without identifiers allow.
"""
import argparse
import glob
import json
import os

KEYS = ("auroc_all", "auroc_three", "auroc_chr21", "auroc_chr22", "auroc_chrX")


def aggregates(d):
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "window-*.json"))):
        s = os.path.basename(path)[len("window-"):-len(".json")]
        out[s] = json.load(open(path))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new-score-key", default="score_window")
    ap.add_argument("--old-score-key", default="score")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="largest |difference| that still counts as agreement. "
                         "0 means the values must be identical")
    args = ap.parse_args()

    new, old = aggregates(args.new), aggregates(args.old)
    shared = sorted(set(new) & set(old))
    print(f"  new {len(new)} run   old {len(old)} run   共通 {len(shared)}")
    if not shared:
        raise SystemExit("no run in common; check the two directories")

    bad, n_cmp = [], 0
    print("\n  === 集計値 ===")
    for s in shared:
        for k in KEYS:
            if k not in new[s] or k not in old[s]:
                continue
            n_cmp += 1
            d = abs(new[s][k] - old[s][k])
            if d > args.tolerance:
                bad.append((s, k, old[s][k], new[s][k], d))
    print(f"  {n_cmp} 値を比べ、許容 {args.tolerance} を超えたのは {len(bad)} 件")
    for s, k, o, n, d in bad[:20]:
        print(f"    {s:36s} {k:12s} 旧 {o:.6f}  新 {n:.6f}  差 {d:.2e}")

    ck = [(s, old[s].get("checkpoint_step"), new[s].get("checkpoint_step"))
          for s in shared]
    diff_ck = [t for t in ck if t[1] is not None and t[2] is not None and t[1] != t[2]]
    known = [t for t in ck if t[1] is not None and t[2] is not None]
    print(f"  採用 checkpoint: 両方に記録のある {len(known)} 本のうち "
          f"食い違い {len(diff_ck)} 本")
    for s, o, n in diff_ck:
        print(f"    {s:36s} 旧 step {o}  新 step {n}")

    print("\n  === 変異 1 件ごとの値 ===")
    rows_cmp, rows_bad, worst, misaligned, pairs = 0, 0, 0.0, 0, 0
    joined_on = None
    for s in shared:
        fn = os.path.join(args.new, f"window-{s}-predictions.jsonl")
        fo = os.path.join(args.old, f"window-{s}-predictions.jsonl")
        if not (os.path.exists(fn) and os.path.exists(fo)):
            continue
        pairs += 1
        with open(fn) as a, open(fo) as b:
            for ln, (x, y) in enumerate(zip(a, b), start=1):
                rn, ro = json.loads(x), json.loads(y)
                if args.old_score_key not in ro:
                    break
                if joined_on is None:
                    joined_on = "vcv_id" if "vcv_id" in ro else "行番号"
                if "vcv_id" in ro and ro["vcv_id"] != rn["vcv_id"]:
                    misaligned += 1
                    continue
                if (ro.get("chrom") != rn.get("chrom")
                        or ro.get("label_pathogenic") != rn.get("label_pathogenic")):
                    misaligned += 1
                    continue
                rows_cmp += 1
                d = abs(rn[args.new_score_key] - ro[args.old_score_key])
                worst = max(worst, d)
                if d > args.tolerance:
                    rows_bad += 1
    if rows_cmp:
        print(f"  {pairs} 本・{rows_cmp:,} 行を {joined_on} で突き合わせ、"
              f"許容 {args.tolerance} を超えたのは {rows_bad:,} 行  "
              f"最大 |差| {worst:.3e}")
        if misaligned:
            print(f"  染色体かラベルが合わず外した行 {misaligned:,}")
    else:
        print("  突き合わせられる行が無い（旧の出力に変異ごとの値が無い）")

    if bad or diff_ck or rows_bad or misaligned:
        raise SystemExit(1)
    print("\n問題なし")


if __name__ == "__main__":
    main()
