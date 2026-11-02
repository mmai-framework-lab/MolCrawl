"""Do the downstream scorings on disk carry their per-variant values?

Twice a campaign has had to score 21 runs again because an output held only an
aggregate: once because a bootstrap interval cannot be resampled from a mean,
and once because the rows that replaced it carried a bare score, with nothing
to join a row to a variant and no way to re-sum it over a narrower range.

This asks the question of the files rather than of the code: for each directory
given, every per-variant file is read, its columns are compared against the
ones required, and its rows are counted. It exits non-zero when a file is
missing a column or a directory has no per-variant file at all, so a job script
can refuse to report a number that cannot be traced back to rows.

  python scripts/check_per_variant_outputs.py \
      --dir <results>/clinvar-zeroshot-gpt2-ep9 --pattern '*/predictions.jsonl' \
      --require vcv_id,chrom,pos,ref_allele,alt_allele,clinical_significance \
      --require label_pathogenic --expect-rows 21350
"""
import argparse
import glob
import json
import os
import sys

IDENTITY = ("vcv_id", "chrom", "pos", "ref_allele", "alt_allele",
            "clinical_significance", "label_pathogenic")


def columns(path):
    """The keys of the first line, which every line of a jsonl file shares."""
    with open(path) as handle:
        first = handle.readline()
    if not first.strip():
        return []
    return list(json.loads(first))


def rows(path):
    with open(path) as handle:
        return sum(1 for line in handle if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True,
                    help="a results directory; repeat for several")
    ap.add_argument("--pattern", default="*/predictions.jsonl",
                    help="where the per-variant files sit inside each directory")
    ap.add_argument("--require", action="append", default=[],
                    help="comma-separated column names; repeat for several")
    ap.add_argument("--expect-rows", type=int, default=None)
    ap.add_argument("--expect-files", type=int, default=None)
    args = ap.parse_args()

    required = [c.strip() for group in (args.require or [",".join(IDENTITY)])
                for c in group.split(",") if c.strip()]
    bad = 0
    for directory in args.dir:
        found = sorted(glob.glob(os.path.join(directory, args.pattern)))
        print(f"\n=== {directory}  ({len(found)} 件) ===")
        if not found:
            print("  ** 変異ごとの値が 1 件も無い **")
            bad += 1
            continue
        if args.expect_files is not None and len(found) != args.expect_files:
            print(f"  ** {args.expect_files} 件のはずが {len(found)} 件 **")
            bad += 1
        for path in found:
            cols = columns(path)
            missing = [c for c in required if c not in cols]
            n = rows(path)
            short = os.path.relpath(path, directory)
            if missing:
                print(f"  ** {short}: 欠けている列 {','.join(missing)} **")
                bad += 1
            elif args.expect_rows is not None and n != args.expect_rows:
                print(f"  ** {short}: {args.expect_rows:,} 行のはずが {n:,} 行 **")
                bad += 1
            else:
                print(f"  {short}: {n:,} 行 / {len(cols)} 列")

    print(f"\n{'問題なし' if not bad else f'問題 {bad} 件'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
