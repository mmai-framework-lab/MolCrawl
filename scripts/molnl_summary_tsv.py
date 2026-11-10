#!/usr/bin/env python3
"""One row per run: the table to hand over.

The per-evaluation-point table is 16,338 rows and is the raw material, not something a
reader opens. This reduces it to what a result is made of -- where a run stopped against
where it was meant to stop, what it reached, how that compares to the arm's seed spread,
and what it cost.

It used to write the retrieval measurements beside it in a table of their own. The
2026-11-10 judgment deletes those measurements in 3.3, so that table and the argument it
read are gone; the test-split round has its own sheet in
scripts/molnl_smoltest_summary_tsv.py.

Values the figures use are computed the same way here: the mean of the last ten
evaluations rather than the best, since the best is the minimum over however many
evaluations a run happened to log.

    python scripts/molnl_summary_tsv.py --table <all-curves tsv> --sizes <json> \
        --bert-root <dir> --gpt2-roots <dir>... --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

PF_DAY = 8.64e19
TOKENS_PER_STEP = 2_621_440
TRAIN_TOKENS = 325_752_832
WINDOW = 10
# molnl valid split, [MASK] rate 0.2, on training_ready_hf_dataset_shuffled:
# report/mlm-baselines-20261102/full/molnl.json (unigram, fit on 200,000 train
# rows and scored on the 1,631 valid rows). The figures' y axis is the valid
# eval_loss_mask, so the floor has to be the valid one too. The 3.8638 that stood
# here was measured on the train split (job 22503, 2026-08-12) and sits 0.78 below.
# The GPT-2 value was already measured on valid.
FLOOR = {"bert": 4.6449, "gpt2": 4.6514}


def read_table(path):
    val, trn, meta = defaultdict(dict), defaultdict(dict), {}
    for row in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
        if row["metric"] == "val_loss_full_split":
            continue
        (val if row["split"] == "validation" else trn)[row["run"]][int(row["step"])] = \
            float(row["value"])
        # The row kept here describes the validation series, which is what every
        # summarised value below is read from; taking whichever row came last labelled
        # those values with the training metric and its resampled mask.
        if row["split"] == "validation":
            meta[row["run"]] = row
    return val, trn, meta


def planned_steps(run, roots):
    """What the run was told to do, from its own manifest. Blank when it wrote none."""
    for root in roots:
        path = os.path.join(root, run, "run_manifest.json")
        if os.path.exists(path):
            sched = json.load(open(path)).get("schedule", {})
            return sched.get("max_steps") or sched.get("max_iters") or ""
    return ""


def window_mean(points):
    steps = sorted(points)[-WINDOW:]
    return sum(points[s] for s in steps) / len(steps)


def spread_of(val, run, meta):
    """The seed spread of the arm this run belongs to, when three seeds exist and held."""
    base = run.replace("_seed1", "").replace("_seed17", "")
    kept = [window_mean(val[n]) for n in (base, base + "_seed1", base + "_seed17")
            if n in val and window_mean(val[n]) < 1.0]
    return (max(kept) - min(kept)) if len(kept) >= 2 else ""


def runs_table(val, trn, meta, sizes, roots, out):
    by_size = {(v["arch"], k.split("_")[1]): v["non_embedding"] for k, v in sizes.items()}
    columns = ["run", "arch", "size", "lr", "seed", "metric", "eval_set", "planned_steps",
               "last_step", "finished", "window_mean", "best", "best_step", "final",
               "train_final", "seed_spread", "tokens_seen", "epochs", "pfdays",
               "non_embedding_params", "floor", "state", "job"]
    written = 0
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\t".join(columns) + "\n")
        for run in sorted(val):
            points, row = val[run], meta[run]
            arch, size = row["arch"], row["size"]
            last = max(points)
            best_step = min(points, key=lambda s: points[s])
            n = by_size.get((arch, size), "")
            tokens = last * TOKENS_PER_STEP
            plan = planned_steps(run, roots)
            floor = FLOOR[arch]
            final = points[last]
            # ASCII only in the data files: these go out as attachments and are opened
            # in whatever a reader has, where a stray encoding turns a Japanese column
            # into mojibake. The prose that interprets them is in the report.
            state = ("diverged" if final >= floor - 0.005 else
                     "finished" if plan and last >= int(plan) else "stopped_short")
            fh.write("\t".join(str(v) for v in [
                run, arch, size, row["lr"], row["seed"], row["metric"], row["eval_set"],
                plan, last, "yes" if plan and last >= int(plan) else "no",
                f"{window_mean(points):.6f}", f"{points[best_step]:.6f}", best_step,
                f"{final:.6f}",
                f"{trn[run][max(trn[run])]:.6f}" if trn.get(run) else "",
                (f"{spread_of(val, run, meta):.6f}"
                 if isinstance(spread_of(val, run, meta), float) else ""),
                tokens, f"{tokens / TRAIN_TOKENS:.2f}",
                f"{6 * n * tokens / PF_DAY:.4f}" if n else "", n, floor, state, row["job"],
            ]) + "\n")
            written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", required=True)
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="the run roots, for each run's own manifest")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    val, trn, meta = read_table(a.table)
    sizes = json.load(open(a.sizes))
    runs_out = os.path.join(a.out_dir, "molnl-summary-runs.tsv")
    print(f"  {runs_table(val, trn, meta, sizes, a.roots, runs_out)} runs -> {runs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
