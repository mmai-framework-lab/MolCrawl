#!/usr/bin/env python3
"""Compare two builds of one corpus: same content, or a different evaluation set?

molecule_nat_lang was repacked with a pre-packing shuffle, and the runs before and after
that rebuild are reported side by side. If the rebuild also moved documents between
train, valid and test, those runs were scored on different data and the numbers are not
comparable (2026-09-25 order §6.1).

The packing makes this less obvious than it sounds: rows are 1,024-token blocks, not
documents, so shuffling before packing changes which documents share a row even when
every document stays in the split it was in. So three things are compared per split:

  rows          how many, and how many are byte-identical between the builds.
  tokens        the total, and the per-token counts. Identical counts with different
                rows means the same text, packed differently.
  types         how many distinct token ids, as a cheap check on the above.
  drift         how far apart the two token distributions are, as the share of the
                split's tokens that would have to move to make them equal.

Same token counts and no identical rows is a repack. A drift of a few parts in ten
thousand is a boundary effect -- packing puts a different handful of documents either
side of the last row. A drift of percent is a different set of documents, and the runs
before and after were scored on different data.

    python scripts/molnl_compare_builds.py --a <dir> --b <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import zlib


def read_split(split, column):
    counts = collections.Counter()
    hashes = set()
    rows = 0
    for row in split:
        ids = row[column]
        counts.update(ids)
        hashes.add(zlib.crc32(bytes(str(ids), "ascii")))
        rows += 1
    return counts, hashes, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="the earlier build")
    ap.add_argument("--b", required=True, help="the later build")
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--column", default="input_ids")
    ap.add_argument("--out")
    a = ap.parse_args()

    from datasets import load_from_disk

    left, right = load_from_disk(a.a), load_from_disk(a.b)
    report = {"a": os.path.abspath(a.a), "b": os.path.abspath(a.b), "splits": {}}
    for name in a.splits.split(","):
        if name not in left or name not in right:
            report["splits"][name] = {"missing_from": [k for k, d in (("a", left), ("b", right))
                                                       if name not in d]}
            continue
        ca, ha, ra = read_split(left[name], a.column)
        cb, hb, rb = read_split(right[name], a.column)
        moved = sum(abs(ca[t] - cb.get(t, 0)) for t in set(ca) | set(cb)) / 2
        total = max(sum(ca.values()), 1)
        report["splits"][name] = {
            "rows": {"a": ra, "b": rb},
            "token_drift": {
                "tokens_that_would_have_to_move": int(moved),
                "share_of_split": round(moved / total, 6),
            },
            "identical_rows": len(ha & hb),
            "tokens": {"a": sum(ca.values()), "b": sum(cb.values())},
            "token_counts_identical": ca == cb,
            "types": {"a": len(ca), "b": len(cb)},
            "reading": ("identical build" if ha == hb else
                        "same tokens, packed differently" if ca == cb else
                        "a boundary effect: under 0.1% of the split's tokens differ"
                        if moved / total < 0.001 else
                        "the two builds hold different text"),
        }
    print(json.dumps(report, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
