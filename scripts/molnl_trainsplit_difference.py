#!/usr/bin/env python3
"""How big is the gap between the pairs whose molecule is in train and the pairs' rest?

The 2026-11-10 judgment's 3.2.1 asks for the two sides and, if they differ, for the
difference to be written. 102 queries against 815 is a small side against a large one, so
a difference of a few points there is not the same news as the same difference on 1,000,
and the report should not have to carry a judgment of that made by eye.

For each of the 14 frames this prints, for Recall@1:

  hits        the counts, which the recalls were rounded from
  difference  in-train minus the rest
  p           the exact two-sided Fisher probability of a table at least this lopsided,
              with the two sides as the two rows. Fisher rather than a normal
              approximation because 0 of 102 happens here twice and the approximation is
              worthless at a zero cell.

It reads the json the split run wrote. Nothing is scored and nothing is scanned.

    python scripts/molnl_trainsplit_difference.py --split-json <json> --out <tsv>
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime

COLUMNS = ("arch", "size", "direction", "hits_in_train", "queries_in_train",
           "recall@1_in_train", "hits_not_in_train", "queries_not_in_train",
           "recall@1_not_in_train", "difference", "p_two_sided_fisher",
           "mrr_in_train", "mrr_not_in_train", "mrr_difference")
ARCH_ORDER = {"gpt2": 0, "bert": 1}
SIZE_ORDER = {"small": 0, "medium": 1, "large": 2, "xl": 3}
DIRECTION_ORDER = {"molecule_to_text": 0, "text_to_molecule": 1}


def log_choose(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_two_sided(a, b, c, d):
    """P of any 2x2 table with these margins no more likely than the observed one."""
    row1, row2 = a + b, c + d
    col1 = a + c
    total = row1 + row2

    def logp(x):
        return (log_choose(row1, x) + log_choose(row2, col1 - x) - log_choose(total, col1))

    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    observed = logp(a)
    terms = [logp(x) for x in range(lo, hi + 1) if logp(x) <= observed + 1e-9]
    top = max(terms)
    return min(1.0, math.exp(top + math.log(sum(math.exp(t - top) for t in terms))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    data = json.load(open(a.split_json, encoding="utf-8"))
    by = {}
    for row in data["split_metrics"]:
        by[(row["arch"], row["size"], row["direction"], row["subset"])] = row

    frames = sorted({k[:3] for k in by},
                    key=lambda k: (ARCH_ORDER.get(k[0], 9), SIZE_ORDER.get(k[1], 9),
                                   DIRECTION_ORDER.get(k[2], 9)))
    rows = []
    for frame in frames:
        inside = by[(*frame, "molecule_in_train")]
        rest = by[(*frame, "molecule_not_in_train")]
        a_hits = round(inside["recall@1"] * inside["queries"])
        c_hits = round(rest["recall@1"] * rest["queries"])
        rows.append({
            "arch": frame[0], "size": frame[1], "direction": frame[2],
            "hits_in_train": a_hits, "queries_in_train": inside["queries"],
            "recall@1_in_train": f"{inside['recall@1']:.4f}",
            "hits_not_in_train": c_hits, "queries_not_in_train": rest["queries"],
            "recall@1_not_in_train": f"{rest['recall@1']:.4f}",
            "difference": f"{inside['recall@1'] - rest['recall@1']:+.4f}",
            "p_two_sided_fisher": f"{fisher_two_sided(a_hits, inside['queries'] - a_hits, c_hits, rest['queries'] - c_hits):.4f}",
            "mrr_in_train": f"{inside['mrr']:.4f}",
            "mrr_not_in_train": f"{rest['mrr']:.4f}",
            "mrr_difference": f"{inside['mrr'] - rest['mrr']:+.4f}",
        })

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("#! report-tsv 1\n")
        fh.write("# model: several (columns: arch,size)\n")
        fh.write("# run: molecule_nat_lang retrieval on the SMolInstruct test split, the "
                 "pairs whose molecule occurs in train against the rest\n")
        fh.write(f"# source: {os.path.abspath(a.split_json)}\n")
        fh.write(f"# generated: "
                 f"{datetime.now().astimezone().isoformat(timespec='seconds')}"
                 f" by scripts/molnl_trainsplit_difference.py\n")
        fh.write("# note: p_two_sided_fisher is the exact two-sided Fisher probability for "
                 "the 2x2 table of hits against misses on the two sides\n")
        fh.write("# note: the 83 positions RDKit will not read are in neither side\n")
        fh.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
    print(f"  wrote {a.out}: {len(rows)} rows")
    for row in rows:
        print("   ", "\t".join(str(row[c]) for c in COLUMNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
