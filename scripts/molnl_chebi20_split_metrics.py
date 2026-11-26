#!/usr/bin/env python3
"""Hits@1, Hits@10, MRR and Mean Rank of the ChEBI-20 runs, split by pretraining overlap.

The 2026-11-26 order, item 6: part of the test split is in the molnl pretraining corpus.
This splits each run's test descriptions by the class scripts/molnl_chebi20_overlap.py
gave them and recomputes the four metrics per group, from the combined ranks table that
scripts/molnl_chebi20_report.py wrote. The candidate set is the same 33,010 in every
group; only the queries are split.

Groups: all; pair (description and molecule in the same corpus record); not_pair (every
other class); and each class on its own.

    python scripts/molnl_chebi20_split_metrics.py --ranks <rt1-...-ranks-*.tsv> \
        --overlap-items <tsv> --out <tsv>
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone

CLASSES = ["pair", "description_and_molecule_apart", "description_only", "molecule_only", "neither"]


def read_tsv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader((line for line in fh if not line.startswith("#")),
                                   delimiter="\t"))


def metrics(r):
    n = len(r)
    return {"n_queries": n, "hits@1": sum(x <= 1 for x in r) / n,
            "hits@10": sum(x <= 10 for x in r) / n, "mrr": sum(1 / x for x in r) / n,
            "mean_rank": sum(r) / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--overlap-items", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cls = {r["cid"]: r["class"] for r in read_tsv(a.overlap_items)}
    unknown = set(cls.values()) - set(CLASSES)
    if unknown:
        raise SystemExit(f"unknown overlap classes {sorted(unknown)}")
    by_arm = defaultdict(list)
    arms = []
    for r in read_tsv(a.ranks):
        if r["arm"] not in by_arm:
            arms.append(r["arm"])
        by_arm[r["arm"]].append((r["description_id"], int(r["rank"])))

    rows = []
    for arm in arms:
        items = by_arm[arm]
        ids = [i for i, _ in items]
        if sorted(ids) != sorted(cls):
            raise SystemExit(f"{arm}: test descriptions differ from the overlap table")
        groups = {"all": [x for _, x in items],
                  "pair": [x for i, x in items if cls[i] == "pair"],
                  "not_pair": [x for i, x in items if cls[i] != "pair"]}
        for c in CLASSES:
            if c != "pair":
                groups[c] = [x for i, x in items if cls[i] == c]
        for g, r in groups.items():
            if not r:
                continue
            m = metrics(r)
            rows.append({"arm": arm, "group": g, "n_queries": m["n_queries"],
                         "hits@1": f"{m['hits@1']:.4f}", "hits@10": f"{m['hits@10']:.4f}",
                         "mrr": f"{m['mrr']:.4f}", "mean_rank": f"{m['mean_rank']:.2f}"})

    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("#! report-tsv 1\n")
        fh.write("# model: the eight runs of rt1-molnl-chebi20-arms (column arm)\n")
        fh.write("# source: rt1-molnl-chebi20-ranks and the per-item overlap table of "
                 "scripts/molnl_chebi20_overlap.py\n")
        fh.write("# note: queries split by pretraining overlap class; candidates are all 33,010 "
                 "in every group. pair = description and molecule in the same corpus record; "
                 "not_pair = every other class\n")
        fh.write(f"# generated: {stamp} by scripts/molnl_chebi20_split_metrics.py\n")
        cols = list(rows[0])
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    for r in rows:
        if r["group"] in ("all", "pair", "not_pair"):
            print("\t".join(str(r[c]) for c in ("arm", "group", "n_queries", "hits@1",
                                                 "hits@10", "mrr", "mean_rank")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
