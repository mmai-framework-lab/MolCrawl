"""Account for every cell the cell-type probe sees, from the JSONL it reads.

The totals in circulation do not add up on their own: 193,108 cells overall
against 134,531 train and 58,531 scored leaves 46 unexplained. The gap is not a
rounding artefact and not a second split -- it is the classes the evaluator sets
aside on the test side before the macro average, which is a different number
from the test side's size. This counts all four quantities from the same file so
they can be stated together.

The counts come from the JSONL the probe is pointed at, not from a report, so a
rebuild of the evaluation set changes them here as well.

The TSV follows the report column convention (docs/07-reports/report_tsv_columns.md):
the name begins with ``rt1-``, the first line declares the version, and the
provenance names the source and when it was written. ``checkpoint`` is left out
rather than filled with a placeholder -- this table counts an evaluation set and
belongs to no checkpoint, which R15 prefers over claiming something unknown.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import sys

CONVENTION_VERSION = 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True, help="the evaluation JSONL the probe reads")
    ap.add_argument("--split-field", default="split")
    ap.add_argument("--label-field", default="cell_type")
    ap.add_argument("--test-label", default="test")
    ap.add_argument(
        "--min-cells-per-label",
        type=int,
        default=20,
        help="classes under this on the test side are set aside before the macro average",
    )
    ap.add_argument(
        "--out-tsv",
        default=None,
        help=(
            "where to write the TSV. The basename must begin with "
            f"'rt{CONVENTION_VERSION}-': the viewer ingests only files that claim "
            "the convention in their name, and a file that claims it in its "
            "contents but not its name is reported as a disagreement (R13)."
        ),
    )
    a = ap.parse_args(argv)

    per_split = collections.Counter()
    test_labels = collections.Counter()
    train_labels = collections.Counter()
    total = 0
    for line in open(a.jsonl, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        total += 1
        split = str(row.get(a.split_field, ""))
        label = str(row.get(a.label_field, ""))
        per_split[split] += 1
        (test_labels if split == a.test_label else train_labels)[label] += 1

    n_train = sum(v for k, v in per_split.items() if k != a.test_label)
    n_test = per_split[a.test_label]
    excluded = {k: v for k, v in test_labels.items() if v < a.min_cells_per_label}
    n_excluded = sum(excluded.values())
    n_scored = n_test - n_excluded

    print(f"JSONL                     {a.jsonl}")
    print(f"細胞 合計                 {total:,}")
    for k in sorted(per_split):
        print(f"  split={k or '(空)':<18s}  {per_split[k]:,}")
    print(f"学習セット                {n_train:,}")
    print(f"テストセット              {n_test:,}")
    print(f"  うち採点から外した細胞  {n_excluded:,}  ({len(excluded)} クラス)")
    for k in sorted(excluded):
        print(f"    {k:<42s} {excluded[k]:,}")
    print(f"  採点した細胞            {n_scored:,}")
    print(f"クラス数 全体 {len(set(test_labels) | set(train_labels)):,}  "
          f"test 側 {len(test_labels):,}  採点 {len(test_labels) - len(excluded):,}")
    print()
    print(f"検算  学習 {n_train:,} + テスト {n_test:,} = {n_train + n_test:,}  "
          f"（合計との差 {total - n_train - n_test}）")
    print(f"検算  テスト {n_test:,} - 外した {n_excluded:,} = 採点 {n_scored:,}")

    if a.out_tsv:
        prefix = f"rt{CONVENTION_VERSION}-"
        if not os.path.basename(a.out_tsv).startswith(prefix):
            raise SystemExit(
                f"--out-tsv must name a file beginning with {prefix!r}; got "
                f"{os.path.basename(a.out_tsv)!r}. See docs/07-reports/report_tsv_columns.md"
            )
        now = datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=9))
        ).isoformat(timespec="seconds")
        n_classes_all = len(set(test_labels) | set(train_labels))
        # LF endings and no BOM: a TSV carries neither (R5, R6). The reports do
        # carry a BOM, which is why this is stated rather than left to the default.
        with open(a.out_tsv, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"#! report-tsv {CONVENTION_VERSION}\n")
            fh.write("# model: not applicable (cell counts of the rna cell-type evaluation set)\n")
            fh.write(f"# source: {os.path.abspath(a.jsonl)}\n")
            fh.write(f"# generated: {now} by scripts/rna_celltype_cell_accounting.py\n")
            fh.write(
                f"# note: subset=test_scored is what accuracy and f1_macro are computed over; "
                f"subset=test is the test side's own size. They differ by the classes under "
                f"{a.min_cells_per_label} cells on the test side, which are listed as "
                f"subset=excluded_label.\n"
            )
            fh.write("subset\tcells\tclasses\tnote\n")
            fh.write(f"total\t{total}\t{n_classes_all}\tevery line of the JSONL\n")
            fh.write(f"train\t{n_train}\t{len(train_labels)}\tsplit != {a.test_label}\n")
            fh.write(f"test\t{n_test}\t{len(test_labels)}\tsplit == {a.test_label}\n")
            fh.write(f"test_excluded\t{n_excluded}\t{len(excluded)}\t"
                     f"classes under {a.min_cells_per_label} cells on the test side\n")
            fh.write(f"test_scored\t{n_scored}\t{len(test_labels) - len(excluded)}\t"
                     f"cells accuracy and f1_macro are computed over\n")
            for k in sorted(excluded):
                fh.write(f"excluded_label\t{excluded[k]}\t1\t{k}\n")
        print(f"wrote {a.out_tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
