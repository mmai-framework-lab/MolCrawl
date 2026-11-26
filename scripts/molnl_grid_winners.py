"""Extract which learning-rate multiplier won each cell of the molnl batch grid.

The 2026-11-26 answer to the 2026-11-24 report (§8.2) extends the grid upwards in
learning rate, but only in the cells where the rate that won was the largest one
that cell tried. Which multiplier won is a measurement, not a rule, so it cannot
be derived: it is read off the finished grid's own per-arm summary.

This writes that reading out as a committed JSON, so the derivation in
``molnl_batch_sweep_arms.py`` can hold the same numbers as a literal and compare
the two at import. The literal is what the arms are built from; this file is where
the numbers came from. A disagreement between them stops the derivation rather than
being resolved one way or the other.

    molnl_grid_winners.py --summary rt1-molnl-bsweep-arm-summary-<date>.tsv \
        --out scripts/molnl_grid_winners_<date>.json

Read-only with respect to the runs: it reads one TSV and writes one JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

#: Only the BERT half runs several rates per batch, so only it has a winner to read.
ARCH = "bert"


def read_summary(path: str):
    """The arm-summary rows, with the report-tsv provenance lines skipped."""
    with open(path, encoding="utf-8") as handle:
        body = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(body, delimiter="\t"))


def winners(rows):
    cells = defaultdict(list)
    for row in rows:
        if row["arch"] != ARCH:
            continue
        cells[(row["size"], int(row["batch"]))].append(
            (float(row["lr_multiplier"]), float(row["best_value"]), float(row["lr"]),
             row["collapsed"]))
    out = []
    for (size, batch), tried in sorted(cells.items()):
        tried.sort()
        multipliers = [m for m, _, _, _ in tried]
        best = min(tried, key=lambda t: t[1])
        out.append({
            "arch": ARCH, "size": size, "batch": batch,
            "multipliers_tried": multipliers,
            "winning_multiplier": best[0],
            "winning_lr": f"{best[2]:.2e}",
            "winning_value": f"{best[1]:.6f}",
            "winner_is_the_largest_tried": best[0] == max(multipliers),
            "collapsed": best[3],
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", required=True,
                    help="an rt1- arm-summary TSV written by molnl_batch_sweep_curves")
    ap.add_argument("--out", required=True, help="the JSON to write")
    args = ap.parse_args(argv)

    rows = read_summary(args.summary)
    cells = winners(rows)
    in_scope = [c for c in cells if c["winner_is_the_largest_tried"]]

    payload = {
        "judged_metric": "eval_loss_mask",
        "rule": "the cell's lowest best_value; a cell is in scope for the 2026-11-26 "
                "answer's item 8.2 when that arm's multiplier is the largest the cell "
                "tried",
        "source_basename": os.path.basename(args.summary),
        "arms_read": sum(1 for r in rows if r["arch"] == ARCH),
        "cells": cells,
        "cells_in_scope": len(in_scope),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"=== {payload['arms_read']} {ARCH} arms over {len(cells)} cells, "
          f"from {payload['source_basename']} ===")
    for cell in cells:
        mark = "in scope" if cell["winner_is_the_largest_tried"] else "-"
        print(f"  {cell['size']:6s} b{cell['batch']:<5d} tried "
              f"{cell['multipliers_tried']}  won x{cell['winning_multiplier']} "
              f"at {cell['winning_lr']} = {cell['winning_value']}  {mark}")
    print(f"\n{len(cells)} cells -> {args.out}  ({len(in_scope)} in scope)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
