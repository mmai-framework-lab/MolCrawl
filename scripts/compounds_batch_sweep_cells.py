#!/usr/bin/env python
"""Per-cell and per-arm tables for the compounds batch-size grid.

The grid runs several rates in each (size, batch) cell and keeps the lowest. Two
questions about that need the whole cell, not the winner:

* **Where in the cell's rates did the winner sit?** A cell whose best value came
  from the highest rate it tried may have its optimum above the range that was
  swept; one whose best came from the lowest may have it below. Only the rates
  that were actually run can answer this, so the cell -- including the arms the
  collapse detector stopped -- is the unit.
* **What value represents an arm that never learned?** The series minimum is the
  right summary for an arm that converged, and the wrong one for an arm whose
  minimum is a single excursion out of a flat band. For those, the band itself is
  reported: its mean and how far apart its ends are.

Writes two tables and prints the per-size rate-1/rate-2 margin.

    python scripts/compounds_batch_sweep_cells.py \\
        --arms report/rt1-compounds-batch-sweep-arms-DATE.tsv \\
        --summary report/rt1-...-summary-bert-DATE.tsv \\
        --curves report/rt1-...-curves-bert-DATE.tsv \\
        --cells-out report/rt1-compounds-batch-sweep-cells-bert-DATE.tsv \\
        --arm-table-out report/rt1-compounds-batch-sweep-armtable-bert-DATE.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compounds_batch_sweep_spread import (  # noqa: E402
    JUDGED,
    SIZES,
    excluded,
    planned,
    read_rows,
    series,
)
from scripts.report_tsv import stamp, write_tsv  # noqa: E402

BATCHES = ("256", "512", "1024", "2048", "4096")

#: Where the band that represents a never-learning arm starts. Before this the arm
#: is still leaving its initialisation, so the early points describe the warmup and
#: not the level the arm settled at.
BAND_FROM = 0.20

CELL_COLUMNS = [
    "size",
    "batch",
    "arms_in_cell",
    "arms_surviving",
    "arms_collapsed",
    "lrs_tried",
    "lr_lowest",
    "lr_highest",
    "best_lr",
    "best_lr_multiplier",
    "best_lr_rank",
    "lr_position",
    "metric",
    "value",
    "fraction_of_budget",
    "source",
]

ARM_COLUMNS = [
    "size",
    "batch",
    "lr",
    "lr_multiplier",
    "steps_planned",
    "steps_reached",
    "fraction_of_budget",
    "ended",
    "mark",
    "metric",
    "value",
    "value_kind",
    "band_from",
    "band_mean",
    "band_low",
    "band_high",
    "band_spread",
    "band_points",
    "stopped_at_value",
    "source",
]


def multipliers(arm_rows):
    """(size, batch, lr) -> the rate's multiplier on its cell's sqrt-scaled base."""
    return {
        (row["size"], row["batch"], row["lr"]): row.get("lr_multiplier", "")
        for row in arm_rows
        if row.get("arch") in (None, "", "bert")
    }


def band(points, total, start=BAND_FROM):
    """Mean, ends and count of the judged series from ``start`` of the budget on.

    Returns ``None`` when the arm has not run past ``start`` yet: a band summary
    taken from the warmup would describe the warmup.
    """
    kept = [value for step, value in points if total and step / total >= start]
    if not kept:
        return None
    return {
        "mean": sum(kept) / len(kept),
        "low": min(kept),
        "high": max(kept),
        "points": len(kept),
    }


def cell_rows(arm_rows, summary_rows, curve_rows, source):
    """One row per (size, batch), with the winning rate's place among the cell's rates."""
    by_arm = series(curve_rows)
    skip = excluded(summary_rows)
    mult = multipliers(arm_rows)
    out = []
    for size in SIZES:
        for batch in BATCHES:
            cell = [
                row
                for row in summary_rows
                if row["size"] == size and row["batch"] == batch
            ]
            if not cell:
                continue
            # Every rate the cell ran, collapsed ones included: the question is
            # what was swept, not what survived.
            rates = sorted({float(row["lr"]) for row in cell})
            alive = [
                row
                for row in cell
                if (row["size"], row["batch"], row["lr"]) not in skip
            ]
            best = min(alive, key=lambda row: float(row["best_value"])) if alive else None
            if best is None:
                row = {
                    "size": size,
                    "batch": batch,
                    "arms_in_cell": len(cell),
                    "arms_surviving": 0,
                    "arms_collapsed": sum(
                        1 for r in cell if r.get("ended") == "collapse-stop"
                    ),
                    "lrs_tried": len(rates),
                    "lr_lowest": f"{rates[0]:.2e}",
                    "lr_highest": f"{rates[-1]:.2e}",
                    "metric": JUDGED,
                    "source": source,
                }
                out.append(row)
                continue
            key = (size, batch, best["lr"])
            rank = rates.index(float(best["lr"])) + 1
            if rank == 1:
                position = "lowest"
            elif rank == len(rates):
                position = "highest"
            else:
                position = "middle"
            points = by_arm.get(key, [])
            total = planned(summary_rows).get(key, 0)
            frac = (points[-1][0] / total) if points and total else 0.0
            out.append(
                {
                    "size": size,
                    "batch": batch,
                    "arms_in_cell": len(cell),
                    "arms_surviving": len(alive),
                    "arms_collapsed": sum(
                        1 for r in cell if r.get("ended") == "collapse-stop"
                    ),
                    "lrs_tried": len(rates),
                    "lr_lowest": f"{rates[0]:.2e}",
                    "lr_highest": f"{rates[-1]:.2e}",
                    "best_lr": best["lr"],
                    "best_lr_multiplier": mult.get(key, ""),
                    "best_lr_rank": rank,
                    "lr_position": position,
                    "metric": JUDGED,
                    "value": f"{float(best['best_value']):.6f}",
                    "fraction_of_budget": f"{frac:.4f}",
                    "source": source,
                }
            )
    return out


def arm_table_rows(arm_rows, summary_rows, curve_rows, source):
    """One row per BERT arm, saying which value represents it and why."""
    by_arm = series(curve_rows)
    steps = planned(summary_rows)
    skip = excluded(summary_rows)
    mult = multipliers(arm_rows)
    out = []
    for row in summary_rows:
        key = (row["size"], row["batch"], row["lr"])
        total = steps.get(key, 0)
        ended = row.get("ended", "")
        collapsed = ended == "collapse-stop"
        # Excluded but not stopped: the detector never fired, so the arm is still
        # running and its minimum is not where it sits.
        never_learned = key in skip and not collapsed
        entry = {
            "size": row["size"],
            "batch": row["batch"],
            "lr": row["lr"],
            "lr_multiplier": mult.get(key, ""),
            "steps_planned": row["steps_planned"],
            "steps_reached": row["steps_reached"],
            "fraction_of_budget": (
                f"{int(row['steps_reached']) / total:.4f}" if total else ""
            ),
            "ended": ended,
            "mark": "collapsed" if collapsed else ("not-learning" if never_learned else ""),
            "metric": JUDGED,
            "stopped_at_value": row.get("stopped_at_value", ""),
            "source": source,
        }
        if never_learned:
            stats = band(by_arm.get(key, []), total)
            if stats is None:
                entry["value_kind"] = "none"
            else:
                entry["value"] = f"{stats['mean']:.6f}"
                entry["value_kind"] = "band_mean"
                entry["band_from"] = f"{BAND_FROM:.2f}"
                entry["band_mean"] = f"{stats['mean']:.6f}"
                entry["band_low"] = f"{stats['low']:.6f}"
                entry["band_high"] = f"{stats['high']:.6f}"
                entry["band_spread"] = f"{stats['high'] - stats['low']:.6f}"
                entry["band_points"] = stats["points"]
        else:
            entry["value"] = f"{float(row['best_value']):.6f}"
            entry["value_kind"] = "minimum"
        out.append(entry)
    out.sort(key=lambda r: (SIZES.index(r["size"]), int(r["batch"]), float(r["lr"])))
    return out


def margins(cells):
    """(size) -> the two lowest batch values and the gap between them.

    The gap is what says how far a ranking between batches can be trusted, so it
    is printed next to the winner rather than left to be read off the table.
    """
    out = {}
    for size in SIZES:
        scored = [
            (float(row["value"]), row["batch"])
            for row in cells
            if row["size"] == size and row.get("value")
        ]
        if len(scored) < 2:
            continue
        scored.sort()
        out[size] = (scored[0], scored[1], scored[1][0] - scored[0][0])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--curves", required=True, type=Path)
    parser.add_argument("--cells-out", required=True, type=Path)
    parser.add_argument("--arm-table-out", required=True, type=Path)
    parser.add_argument("--spread", type=Path, help="the equal-fraction table, to print the best batch per fraction")
    args = parser.parse_args(argv)

    arm_rows = read_rows(args.arms)
    summary_rows = [row for row in read_rows(args.summary) if row.get("arch") == "bert"]
    curve_rows = read_rows(args.curves)

    cells = cell_rows(arm_rows, summary_rows, curve_rows, str(args.summary))
    arms = arm_table_rows(arm_rows, summary_rows, curve_rows, str(args.summary))

    provenance = {
        "model": "compounds bert batch-size grid",
        "source": f"{args.summary}; {args.curves}; {args.arms}",
        "generated": stamp("scripts/compounds_batch_sweep_cells.py"),
    }
    n_cells = write_tsv(args.cells_out, CELL_COLUMNS, cells, provenance)
    n_arms = write_tsv(args.arm_table_out, ARM_COLUMNS, arms, provenance)

    print("########## cells: where the winning rate sat among the cell's rates ##########")
    print(f"{'size':<7}{'batch':>6}{'rates':>7}{'rank':>6}  {'position':<9}{'best lr':>10}{'x':>5}{'value':>11}{'read':>8}")
    for row in cells:
        print(
            f"{row['size']:<7}{row['batch']:>6}{row['lrs_tried']:>7}"
            f"{row.get('best_lr_rank', ''):>6}  {row.get('lr_position', ''):<9}"
            f"{row.get('best_lr', ''):>10}{row.get('best_lr_multiplier', ''):>5}"
            f"{row.get('value', ''):>11}"
            f"{100 * float(row.get('fraction_of_budget') or 0):>7.1f}%"
        )
    print(f"\n{n_cells} cell row(s) -> {args.cells_out}")

    print("\n########## how far apart the best two batches are ##########")
    for size, (first, second, gap) in margins(cells).items():
        print(
            f"{size:<7} 1st b{first[1]:<5} {first[0]:.6f}   "
            f"2nd b{second[1]:<5} {second[0]:.6f}   gap {gap:.6f}"
        )

    print("\n########## arms that are not represented by their minimum ##########")
    for row in arms:
        if row["mark"]:
            print(
                f"{row['size']:<7} b{row['batch']:<5} {row['lr']:>9} "
                f"{row['mark']:<13} {row['metric']}={row.get('value', ''):<10} "
                f"({row['value_kind']}"
                + (
                    f", band {row['band_low']}-{row['band_high']}, "
                    f"spread {row['band_spread']}, {row['band_points']} points"
                    if row["value_kind"] == "band_mean"
                    else ""
                )
                + ")"
            )
    print(f"\n{n_arms} arm row(s) -> {args.arm_table_out}")

    if args.spread:
        print("\n########## best batch at each fraction of the budget ##########")
        for row in read_rows(args.spread):
            print(
                f"{row['size']:<7}{100 * float(row['fraction_of_budget']):>5.0f}%"
                f"   best b{row['best_batch']:<5} {row['best_of_batches']}"
                f"   spread {row['spread']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
