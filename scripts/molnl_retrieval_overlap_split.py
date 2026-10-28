#!/usr/bin/env python3
"""Split an already scored retrieval measurement by whether the corpus holds the pair.

105 of the 1,000 evaluation pairs occur in the pretraining corpus as one record -- the
molecule and its description on the two sides of the same
`<s>[INST] ... [/INST]...<|endoftext|>`. A score above chance over all 1,000 could be
those 105 and nothing else, so the 2026-10-27 judgment asks for the three figures
separately: all 1,000, the 895 the corpus does not hold, and the 105 it does.

Nothing is scored again here. Each measurement's per-query ranks were written when it ran,
and the queries are numbered in the test-pair order, which is also the order the pair
index is written in, so the split is a partition of rows that already exist.

Recall@1 is tested against chance with an exact binomial tail rather than a normal
approximation: the 105-pair bucket expects one hit at chance, and the approximation is
not worth reading there.

    python scripts/molnl_retrieval_overlap_split.py --ranks-dir <dir> \
        --pair-index <json> --out <tsv>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

PRETRAIN_DATASET = "osunlp/SMolInstruct"
EVAL_DATASET = "zjunlp/Mol-Instructions"
CHANCE_RECALL1 = 0.01
# 1/100 + 1/99 + ... : the mean reciprocal rank of a candidate placed at random.
CHANCE_MRR = round(sum(1.0 / r for r in range(1, 101)) / 100, 4)


def binomial_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p). Exact; n here is at most 1,000."""
    if k <= 0:
        return 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def scored(path):
    """{direction: {query index: rank}} from a measurement's per-query ranks."""
    out = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[row["direction"]][int(row["query"])] = int(row["rank"])
    return out


def bucket(ranks):
    """Recall@1, MRR and the count over one set of queries."""
    n = len(ranks)
    if not n:
        return None
    hits = sum(1 for r in ranks if r == 1)
    return {"pairs": n, "recall@1": hits / n, "hits@1": hits,
            "mrr": sum(1.0 / r for r in ranks) / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ranks-dir", required=True)
    ap.add_argument("--pair-index", required=True)
    ap.add_argument("--wrap", default="inst", help="only measurements asked in this form")
    ap.add_argument("--notation", default="smiles")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    index = json.load(open(a.pair_index, encoding="utf-8"))
    inside = set(index["pairs_in_one_record"])
    print(f"  pairs the corpus holds as one record: {len(inside)} of {index['pairs']}")

    rows = []
    for path in sorted(glob.glob(os.path.join(a.ranks_dir, "retrieval-*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        if d.get("wrap") != a.wrap or d.get("molecule_notation") != a.notation:
            continue
        ranks = scored(path.replace(".json", "-ranks.tsv"))
        for direction, summary in d["directions"].items():
            per_query = ranks[direction]
            if len(per_query) != summary["queries"]:
                raise SystemExit(f"{d['label']} {direction}: {len(per_query)} ranks for "
                                 f"{summary['queries']} queries")
            # The run's own recall is reproduced from the ranks before they are split,
            # so a mismatch in the split cannot be blamed on the ranks.
            whole = bucket(list(per_query.values()))
            if abs(whole["recall@1"] - summary["recall@1"]) > 1e-9:
                raise SystemExit(f"{d['label']} {direction}: ranks give "
                                 f"{whole['recall@1']} against {summary['recall@1']}")
            # Not "1,000 pairs": BERT scores 250 queries in the description->molecule
            # direction, and a row that says 1,000 next to a count of 250 reads as an
            # error. The count stands in the pairs column; the name says which range.
            sets = {
                "全体（採点したクエリ）": list(per_query.values()),
                "895 対（コーパスに無い）": [r for q, r in per_query.items()
                                             if q not in inside],
                "105 対（コーパスに在る）": [r for q, r in per_query.items()
                                             if q in inside],
            }
            for name, got in sets.items():
                b = bucket(got)
                if b is None:
                    continue
                rows.append({
                    "model": d["label"].split("__")[0], "arch": d["arch"],
                    "size": d["size"], "direction": direction, "pair_set": name,
                    "pairs": b["pairs"], "recall@1": round(b["recall@1"], 4),
                    "hits@1": b["hits@1"], "mrr": round(b["mrr"], 4),
                    "chance_recall@1": CHANCE_RECALL1, "chance_mrr": CHANCE_MRR,
                    "recall@1_binomial_p": f"{binomial_tail(b['hits@1'], b['pairs'], CHANCE_RECALL1):.3g}",
                    "format": f"{a.wrap}/{a.notation}",
                    "pretrain_dataset": PRETRAIN_DATASET, "eval_dataset": EVAL_DATASET,
                })

    if not rows:
        raise SystemExit(f"no {a.wrap}/{a.notation} measurement under {a.ranks_dir}")
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {a.out} ({len(rows)} rows, "
          f"{len({(r['model'], r['direction']) for r in rows})} measurements)")
    for r in rows:
        if r["pair_set"].startswith("895"):
            print(f"    {r['model']:22s} {r['direction']:17s} "
                  f"R@1 {r['recall@1']:.4f}  MRR {r['mrr']:.4f}  "
                  f"p={r['recall@1_binomial_p']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
