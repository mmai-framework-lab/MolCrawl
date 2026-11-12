#!/usr/bin/env python3
"""The molnl downstream results: the SMolInstruct test-split retrieval, one value per row.

The 2026-11-12 order asks for the fourteen measurements taken on the test split, and for
the Mol-Instructions round to stay out -- those measurements were deleted under the
2026-11-10 judgment's 3.3, so a file that mixed them in would name files that no longer
exist.

The columns it names decide the shape. ``chance`` and ``p_value`` are single columns, and
both depend on which metric the row carries -- chance Recall@1 is 1/candidates while
chance MRR is H(candidates)/candidates, a factor of five apart -- so a row is one metric
of one measurement, and each of the fourteen scorings contributes one row per metric. The
fourteen on their own are the recall@1 rows.

p is the exact one-sided binomial tail for the hit count under that metric's chance level,
imported from scripts/molnl_smoltest_summary_tsv so that this table cannot disagree with
the sheet already handed over. A rank-based metric has no such test here, so MRR and the
median rank carry an empty cell rather than a number that would be read as a p.

``--adopted-json`` is the output of scripts/molnl_bert_adopted_checkpoint.py. When it is
given, every BERT row's checkpoint is checked against the run's own
``best_model_checkpoint``, which is what the order means by the adopted checkpoint.

    python scripts/molnl_downstream_tsv.py --retrieval-dir <dir> --out <tsv>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime

from molnl_smoltest_summary_tsv import binomial_tail, harmonic

COLUMNS = ("arch", "size", "direction", "metric", "value", "pairs", "candidates",
           "chance", "p_value", "adopted_checkpoint", "source")
RECALLS = (1, 5, 10)
SIZE_ORDER = {"small": 0, "medium": 1, "large": 2, "xl": 3}
ARCH_ORDER = {"gpt2": 0, "bert": 1}
DIRECTION_ORDER = {"molecule_to_text": 0, "text_to_molecule": 1}


def adopted_of(summary):
    """The checkpoint the scoring was given, as a name rather than a path."""
    path = (summary.get("model_path") or "").rstrip("/")
    return os.path.basename(path)


def run_and_checkpoint(summary):
    """<run directory>/<checkpoint>, for the provenance block.

    The column holds the bare name the order asks for, but four GPT-2 rows all reading
    ckpt.pt say nothing about which run each came from, so the provenance carries the run
    as well.
    """
    path = (summary.get("model_path") or "").rstrip("/")
    if not path:
        return ""
    return f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"


def rows_of(path):
    summary = json.load(open(path, encoding="utf-8"))
    source = os.path.basename(path)
    adopted = adopted_of(summary)
    out = []
    for direction, result in summary["directions"].items():
        n = result["queries"]
        candidates = result["candidates"]
        common = {"arch": summary["arch"], "size": summary["size"],
                  "direction": direction, "pairs": n, "candidates": candidates,
                  "adopted_checkpoint": adopted, "source": source,
                  "_run_and_checkpoint": run_and_checkpoint(summary)}
        for k in RECALLS:
            key = f"recall@{k}"
            chance = k / candidates
            hits = round(result[key] * n)
            out.append({**common, "metric": key, "value": f"{result[key]:.4f}",
                        "chance": f"{chance:.4f}",
                        "p_value": f"{binomial_tail(hits, n, chance):.3e}"})
        out.append({**common, "metric": "mrr", "value": f"{result['mrr']:.4f}",
                    "chance": f"{harmonic(candidates) / candidates:.4f}",
                    "p_value": ""})
        out.append({**common, "metric": "median_rank",
                    "value": f"{result['median_rank']:.1f}",
                    "chance": f"{(candidates + 1) / 2:.1f}", "p_value": ""})
    return out


def check_adopted(rows, adopted_json):
    """Does every BERT row name the checkpoint that run's trainer settled on?"""
    if not adopted_json or not os.path.exists(adopted_json):
        return ["the adopted-checkpoint json was not given or does not exist"]
    # Keyed by size, so a row is checked against its own run rather than against the set
    # of all three. bert large's best sits at 23,800 and small's and medium's at 24,000;
    # a membership test would pass a row that named another size's checkpoint.
    best = {}
    for entry in json.load(open(adopted_json, encoding="utf-8")):
        run = os.path.basename((entry.get("run") or "").rstrip("/"))
        path = entry.get("best_model_checkpoint") or ""
        for size in ("small", "medium", "large", "xl"):
            if path and f"_{size}_" in run:
                best[size] = (run, os.path.basename(path.rstrip("/")))
    problems = []
    for row in rows:
        if row["arch"] != "bert":
            continue
        if row["size"] not in best:
            problems.append(f"bert {row['size']}: no run in the adopted json")
            continue
        run, want = best[row["size"]]
        if row["adopted_checkpoint"] != want:
            problems.append(f"bert {row['size']}: scored {row['adopted_checkpoint']}, "
                            f"but {run} settled on {want}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--retrieval-dir", required=True)
    ap.add_argument("--adopted-json", default=None)
    ap.add_argument("--pairs-file", default=None,
                    help="named in the provenance as what was scored")
    ap.add_argument("--job", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows, seen = [], set()
    for path in sorted(glob.glob(os.path.join(a.retrieval_dir, "retrieval-*.json"))):
        for row in rows_of(path):
            key = (row["arch"], row["size"], row["direction"], row["metric"])
            if key in seen:
                raise SystemExit(f"{path}: {key} is already in the table from another "
                                 "file; the same frame must not be scored twice")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise SystemExit(f"no retrieval-*.json under {a.retrieval_dir}")

    problems = check_adopted(rows, a.adopted_json)
    if problems:
        raise SystemExit("the adopted checkpoints do not agree: " + "; ".join(problems))

    frames = {(r["arch"], r["size"], r["direction"]) for r in rows}
    print(f"  {len(frames)} measurements, {len(rows)} rows "
          f"({len(rows) // max(len(frames), 1)} metrics each)", flush=True)
    if a.adopted_json:
        print("  every BERT row names its run's best_model_checkpoint", flush=True)

    rows.sort(key=lambda r: (ARCH_ORDER.get(r["arch"], 9), SIZE_ORDER.get(r["size"], 9),
                             DIRECTION_ORDER.get(r["direction"], 9),
                             list(("recall@1", "recall@5", "recall@10", "mrr",
                                   "median_rank")).index(r["metric"])))
    checkpoints = {}
    for row in rows:
        checkpoints.setdefault(f"{row['arch']} {row['size']}",
                               row["_run_and_checkpoint"] or row["adopted_checkpoint"])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("#! report-tsv 1\n")
        fh.write("# model: several (columns: arch,size)\n")
        fh.write("# run: molecule_nat_lang retrieval on the SMolInstruct test split\n")
        fh.write("# checkpoint: column: adopted_checkpoint\n")
        for frame, name in checkpoints.items():
            fh.write(f"# checkpoint: {frame}: {name}\n")
        if a.pairs_file:
            fh.write(f"# source: pairs {os.path.abspath(a.pairs_file)}\n")
        fh.write(f"# source: {os.path.abspath(a.retrieval_dir)}\n")
        if a.adopted_json:
            fh.write(f"# source: adopted checkpoints {os.path.abspath(a.adopted_json)}\n")
        if a.job:
            fh.write(f"# job: {a.job}\n")
        fh.write(f"# generated: "
                 f"{datetime.now().astimezone().isoformat(timespec='seconds')}"
                 f" by scripts/molnl_downstream_tsv.py\n")
        fh.write("# note: one row is one metric of one measurement; the fourteen "
                 "measurements are the recall@1 rows\n")
        fh.write("# note: chance depends on the metric. recall@k is k/candidates, mrr is "
                 "H(candidates)/candidates = 0.0519 at 100 candidates, median_rank is "
                 "(candidates+1)/2\n")
        fh.write("# note: p_value is the exact one-sided binomial tail for the hit count "
                 "under this row's chance level. mrr and median_rank have no such test "
                 "here and carry an empty cell\n")
        fh.write("# note: the scoring ranks by log p(candidate | query) over the "
                 "candidate's token positions only -- BERT by pseudo-likelihood, GPT-2 by "
                 "causal likelihood. The two architectures' values are not comparable "
                 "with each other; the ranks they induce are\n")
        fh.write("# note: the Mol-Instructions round is not in this file. Those "
                 "measurements were deleted under the 2026-11-10 judgment's 3.3\n")
        fh.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
    print(f"  wrote {a.out}: {len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
