#!/usr/bin/env python
"""How far apart the batches are, read at equal fractions of the budget.

A grid arm's step count is set by its batch, so at any wall-clock moment the arms
of one size sit at different step counts: batch 256 runs 450,000 steps where batch
4096 runs 28,125. Comparing them by date therefore compares different amounts of
training. This reads every arm at the same *fraction of its own budget* instead,
and reports the spread between batches at each fraction, so the table does not
move when it is regenerated on another day.

Reads the curve TSV written by ``compounds_batch_sweep_curves.py`` and writes one
row per (size, fraction).

    python scripts/compounds_batch_sweep_spread.py \
        --curves report/rt1-...-curves-bert-DATE.tsv \
        --summary report/rt1-...-summary-bert-DATE.tsv \
        --out report/rt1-compounds-batch-sweep-spread-bert-DATE.tsv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report_tsv import stamp, write_tsv  # noqa: E402

JUDGED = "eval_loss_mask"
FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00)
SIZES = ("small", "medium", "large", "xl")

COLUMNS = [
    "size",
    "fraction_of_budget",
    "batches_read",
    "arms_read",
    "best_of_batches",
    "worst_of_batches",
    "spread",
    "best_batch",
    "worst_batch",
]


def read_rows(path):
    """The data rows of a report TSV, as dicts, with the ``#`` meta lines dropped."""
    header = None
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            out.append(dict(zip(header, parts)))
    return out


def excluded(summary_rows):
    """Arms whose series must not be read as a point on a batch curve.

    An arm the collapse detector stopped, and an arm that is climbing away from
    its own minimum, are not measuring the batch; they are measuring the rate.
    """
    out = set()
    for row in summary_rows:
        key = (row["size"], row["batch"], row["lr"])
        if row.get("ended") == "collapse-stop":
            out.add(key)
            continue
        best = float(row["best_value"])
        final = float(row["final_value"])
        # climbing: the run is well above the minimum it already reached, but the
        # detector cannot fire because it is still below the degenerate threshold.
        if row.get("ended") == "running" and final - best > 0.1:
            out.add(key)
    return out


def series(curve_rows):
    """(size, batch, lr) -> [(step, value)] of the judged metric, in step order."""
    out = {}
    for row in curve_rows:
        if row["metric"] != JUDGED:
            continue
        key = (row["size"], row["batch"], row["lr"])
        out.setdefault(key, []).append((int(row["step"]), float(row["value"])))
    for key in out:
        out[key].sort()
    return out


def planned(summary_rows):
    return {
        (row["size"], row["batch"], row["lr"]): int(row["steps_planned"])
        for row in summary_rows
    }


def reached(summary_rows):
    """(size, batch, lr) -> the fraction of its budget the arm has actually run.

    Holding a value *before* a fraction is not the same as having reached it: an
    arm at 44 % of its budget has a value at or before 80 %, and reporting it on
    an 80 % row would read as a result it has not produced.
    """
    out = {}
    for row in summary_rows:
        key = (row["size"], row["batch"], row["lr"])
        step = int(row["steps_reached"])
        total = int(row["steps_planned"])
        out[key] = step / total if total else 0.0
    return out


def best_up_to(points, limit):
    """The lowest value at or before ``limit`` steps, or None if nothing is there."""
    seen = [value for step, value in points if step <= limit]
    return min(seen) if seen else None


def rows_for(curve_rows, summary_rows, fractions=FRACTIONS):
    by_arm = series(curve_rows)
    steps = planned(summary_rows)
    run_to = reached(summary_rows)
    skip = excluded(summary_rows)
    out = []
    for size in SIZES:
        arms = [key for key in by_arm if key[0] == size and key not in skip]
        if not arms:
            continue
        for fraction in fractions:
            # Every arm of the size must have run this far. One that has not would
            # otherwise contribute the value it holds from an earlier fraction, and
            # the row would state a comparison nothing has measured yet.
            if any(run_to.get(key, 0.0) + 1e-9 < fraction for key in arms):
                continue
            per_batch = {}
            read = 0
            for key in arms:
                limit = steps[key] * fraction
                value = best_up_to(by_arm[key], limit)
                if value is None:
                    continue
                read += 1
                batch = key[1]
                if batch not in per_batch or value < per_batch[batch]:
                    per_batch[batch] = value
            want = {key[1] for key in arms}
            if per_batch.keys() != want:
                continue
            low = min(per_batch, key=lambda b: per_batch[b])
            high = max(per_batch, key=lambda b: per_batch[b])
            out.append(
                {
                    "size": size,
                    "fraction_of_budget": f"{fraction:.2f}",
                    "batches_read": len(per_batch),
                    "arms_read": read,
                    "best_of_batches": f"{per_batch[low]:.6f}",
                    "worst_of_batches": f"{per_batch[high]:.6f}",
                    "spread": f"{per_batch[high] - per_batch[low]:.6f}",
                    "best_batch": low,
                    "worst_batch": high,
                }
            )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    curve_rows = read_rows(args.curves)
    summary_rows = read_rows(args.summary)
    skip = excluded(summary_rows)
    rows = rows_for(curve_rows, summary_rows)

    print(f"=== {len(summary_rows)} arm(s); {len(skip)} excluded, "
          f"{len(summary_rows) - len(skip)} read ===")
    for key in sorted(skip):
        print(f"  excluded: {key[0]} b{key[1]} lr{key[2]}")
    print()
    header = "size      frac  batches  arms      best     worst    spread  low   high"
    print(header)
    for row in rows:
        print(f"{row['size']:<9}{float(row['fraction_of_budget']) * 100:4.0f}%"
              f"{row['batches_read']:9}{row['arms_read']:6}"
              f"{float(row['best_of_batches']):10.6f}{float(row['worst_of_batches']):10.6f}"
              f"{float(row['spread']):10.6f}  {row['best_batch']:>5} {row['worst_batch']:>5}")

    write_tsv(
        args.out,
        COLUMNS,
        rows,
        {
            "model": "several (columns: size,batch,lr aggregated to size)",
            "source": f"{args.curves.name} and {args.summary.name}",
            "job": os.environ.get("SLURM_JOB_ID") or None,
            "generated": stamp("scripts/compounds_batch_sweep_spread.py"),
            "note": [
                f"each arm read at the same fraction of its own budget; the value is "
                f"the lowest {JUDGED} at or before that fraction",
                "a size appears at a fraction only when every one of its batches has "
                "reached that fraction",
                "arms the collapse detector stopped, and arms climbing more than 0.1 "
                "above their own minimum, are excluded: they measure the rate, not the batch",
            ],
        },
    )
    print(f"\n{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
