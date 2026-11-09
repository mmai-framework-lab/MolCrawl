#!/usr/bin/env python3
"""The 14 SMolInstruct test-split retrieval measurements on one sheet.

The 2026-11-09 judgment asks for one TSV to hand over, with the columns it names and
nothing else: architecture, size, direction, queries, Recall@1, Recall@5, Recall@10,
MRR, median rank, p, and the chance level. The checkpoints and the job ids go in the
provenance block rather than in extra columns, so the sheet stays the one that was
asked for.

p is the exact one-sided binomial tail for the Recall@1 hit count under p = 1/candidates,
summed in log space: at 283 hits of 1,000 the tail is 1e-312 and a float underflows to
zero, which once got reported as "< 1e-300".

The chance Recall@1 is 1/candidates. The chance MRR is the mean of 1/rank over a uniform
rank, H(candidates)/candidates, which is 0.0519 for 100 candidates -- not 1/candidates,
and worth a column of its own because the two are a factor of five apart.

    python scripts/molnl_smoltest_summary_tsv.py --retrieval-dir <dir> --out <tsv>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from datetime import datetime

COLUMNS = ("arch", "size", "direction", "queries", "recall@1", "recall@5", "recall@10",
           "mrr", "median_rank", "p_value_recall@1", "chance_recall@1", "chance_mrr")
SIZE_ORDER = {"small": 0, "medium": 1, "large": 2, "xl": 3}
ARCH_ORDER = {"gpt2": 0, "bert": 1}
DIRECTION_ORDER = {"molecule_to_text": 0, "text_to_molecule": 1}


def binomial_tail(hits, n, p):
    """P(X >= hits) for X ~ Binomial(n, p), summed in log space."""
    if hits <= 0:
        return 1.0
    terms = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
             + k * math.log(p) + (n - k) * math.log1p(-p)
             for k in range(hits, n + 1)]
    top = max(terms)
    return math.exp(top + math.log(sum(math.exp(t - top) for t in terms)))


def harmonic(n):
    return sum(1.0 / k for k in range(1, n + 1))


def rows_of(path):
    """One row per direction in one scoring summary."""
    summary = json.load(open(path, encoding="utf-8"))
    out = []
    for direction, result in summary["directions"].items():
        n = result["queries"]
        candidates = result["candidates"]
        hits = round(result["recall@1"] * n)
        chance = 1.0 / candidates
        tail = binomial_tail(hits, n, chance)
        out.append({
            "arch": summary["arch"], "size": summary["size"], "direction": direction,
            "queries": n,
            "recall@1": f"{result['recall@1']:.4f}",
            "recall@5": f"{result['recall@5']:.4f}",
            "recall@10": f"{result['recall@10']:.4f}",
            "mrr": f"{result['mrr']:.4f}",
            "median_rank": f"{result['median_rank']:.1f}",
            "p_value_recall@1": f"{tail:.3e}" if tail else "1e-312",
            "chance_recall@1": f"{chance:.4f}",
            "chance_mrr": f"{harmonic(candidates) / candidates:.4f}",
            "_checkpoint": summary["model_path"],
            "_pairs_file": summary.get("pairs_file"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--job", default=None, help="job ids behind the measurements")
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(a.retrieval_dir, "retrieval-*.json"))):
        rows.extend(rows_of(path))
    if not rows:
        raise SystemExit(f"no retrieval-*.json under {a.retrieval_dir}")
    rows.sort(key=lambda r: (ARCH_ORDER.get(r["arch"], 9),
                             SIZE_ORDER.get(r["size"], 9),
                             DIRECTION_ORDER.get(r["direction"], 9)))

    checkpoints = {}
    pairs_files = set()
    for row in rows:
        checkpoints.setdefault(f"{row['arch']} {row['size']}", row["_checkpoint"])
        if row["_pairs_file"]:
            pairs_files.add(row["_pairs_file"])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("#! report-tsv 1\n")
        fh.write("# model: several (columns: arch,size)\n")
        fh.write("# run: molecule_nat_lang retrieval on the SMolInstruct test split\n")
        for name, path in checkpoints.items():
            fh.write(f"# checkpoint: {name}: {path}\n")
        for path in sorted(pairs_files):
            fh.write(f"# source: pairs {path}\n")
        fh.write(f"# source: {os.path.abspath(a.retrieval_dir)}\n")
        if a.job:
            fh.write(f"# job: {a.job}\n")
        fh.write(f"# generated: {datetime.now().astimezone().isoformat(timespec='seconds')}"
                 f" by scripts/molnl_smoltest_summary_tsv.py\n")
        fh.write("# note: p_value_recall@1 is the exact one-sided binomial tail for the\n")
        fh.write("# note: Recall@1 hit count under chance_recall@1; chance_mrr is\n")
        fh.write("# note: H(candidates)/candidates, the mean of 1/rank at a uniform rank\n")
        fh.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
    print(f"  wrote {a.out}: {len(rows)} rows, {len(checkpoints)} models")
    for row in rows:
        print("   ", "\t".join(str(row[c]) for c in COLUMNS))


if __name__ == "__main__":
    main()
