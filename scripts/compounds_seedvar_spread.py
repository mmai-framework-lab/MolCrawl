"""Measure the spread of the end-of-run rise across runs that differ only in the seed.

The 2026-11-20 directive's §4 left one number unmeasured. Its question -- whether the
existing 2,560 ladder spent part of its run with the learning rate at the floor -- was
answered no: all four ladder runs decay over exactly their own length. So the +0.5260
rise from minimum to end of the 2,560 large arm, against +0.0999 for the grid's 2,048
large arm, is not the schedule, and what is left between the two runs is a 1.25x
difference in batch carrying a 5.3x difference in the rise.

Whether 5.3x is large depends on how much the rise moves when nothing moves. The grid
cannot say: it has four sizes per batch, not four seeds, so its within-batch agreement
is agreement across sizes. This reads the three runs that differ only in the seed and
reports the spread, beside whatever reference runs are named for comparison.

    compounds_seedvar_spread.py --run seed42=<dir> --run seed1=<dir> ... \
        [--reference existing-2560=<dir>] --out rt1-...tsv

Each run is read from the ``logging_*.csv`` nanoGPT writes beside its checkpoints
(``iter, train_loss, val_loss``), which is the run's own record rather than a scheduler
log that may have been rotated. A run whose series is missing is named and not
silently dropped.

Read-only: it opens logs and writes one TSV.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_tsv import check_name, stamp, write_tsv  # noqa: E402

SEQUENCE_LENGTH = 1024

COLUMNS = ["label", "role", "seed", "run", "batch", "max_iters", "eval_points",
           "metric", "best_value", "best_iter", "best_fraction_of_budget",
           "final_value", "final_iter", "final_minus_best", "commit", "source"]


def series(run_dir):
    """(iter, val_loss) from the run's own logging CSV, in file order."""
    paths = sorted(glob.glob(os.path.join(run_dir, "logging_*.csv")))
    points, used = [], []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                continue
            used.append(os.path.basename(path))
            for row in reader:
                if len(row) < 3 or not row[0].strip():
                    continue
                try:
                    points.append((int(row[0]), float(row[2])))
                except ValueError:
                    continue
    return points, used


def manifest_facts(run_dir):
    """seed, batch, max_iters and the commit, from the run's own manifest."""
    path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(path):
        return {}
    blob = json.load(open(path, encoding="utf-8"))
    return {
        "seed": blob.get("seed", {}).get("seed"),
        "batch": blob.get("batch", {}).get("effective_global_batch"),
        "max_iters": blob.get("schedule", {}).get("max_iters"),
        "commit": blob.get("run", {}).get("git", {}).get("commit"),
    }


def row_for(label, role, run_dir):
    points, used = series(run_dir)
    if not points:
        return None
    facts = manifest_facts(run_dir)
    best_val, best_iter = min((v, i) for i, v in points)
    final_iter, final_val = points[-1]
    planned = facts.get("max_iters") or final_iter
    return {
        "label": label, "role": role,
        "seed": facts.get("seed"), "run": os.path.basename(run_dir),
        "batch": facts.get("batch"), "max_iters": facts.get("max_iters"),
        "eval_points": len(points), "metric": "val_loss",
        "best_value": f"{best_val:.6f}", "best_iter": best_iter,
        "best_fraction_of_budget": f"{best_iter / planned:.4f}",
        "final_value": f"{final_val:.6f}", "final_iter": final_iter,
        "final_minus_best": f"{final_val - best_val:+.6f}",
        "commit": facts.get("commit"),
        "source": " ".join(used),
    }


def pairs(values):
    out = []
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"expected label=path, got {item!r}")
        label, _, path = item.partition("=")
        out.append((label, path))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", metavar="LABEL=DIR",
                    help="a run of the spread measurement; repeatable")
    ap.add_argument("--reference", action="append", metavar="LABEL=DIR",
                    help="a run read for comparison only, kept out of the spread")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    check_name(args.out)

    rows, missing = [], []
    for role, items in (("spread", pairs(args.run)),
                        ("reference", pairs(args.reference))):
        for label, path in items:
            row = row_for(label, role, path)
            if row is None:
                missing.append((label, path))
                continue
            rows.append(row)

    if not rows:
        raise SystemExit("no run had an evaluation series")

    for label, path in missing:
        print(f"  ** no evaluation series for {label} ({path})")

    print(f"\n{'label':16s} {'role':9s} {'seed':>4s} {'batch':>5s} {'points':>6s} "
          f"{'best':>9s} {'at iter':>8s} {'of budget':>9s} {'final':>9s} "
          f"{'final-best':>10s}")
    for row in rows:
        print(f"{row['label']:16s} {row['role']:9s} {str(row['seed']):>4s} "
              f"{str(row['batch']):>5s} {row['eval_points']:>6d} "
              f"{row['best_value']:>9s} {row['best_iter']:>8,} "
              f"{float(row['best_fraction_of_budget'])*100:>8.1f}% "
              f"{row['final_value']:>9s} {row['final_minus_best']:>10s}")

    spread = [float(r["final_minus_best"]) for r in rows if r["role"] == "spread"]
    notes = [
        "final_minus_best = final_value - best_value over the run's own evaluation "
        "points; best_fraction_of_budget = best_iter / max_iters",
        "role=spread are the runs that differ only in the seed; role=reference are "
        "read for comparison and are not part of the spread",
    ]
    if len(spread) >= 2:
        low, high = min(spread), max(spread)
        width = high - low
        print(f"\n=== the spread over {len(spread)} runs that differ only in the seed ===")
        print(f"  final - best: {low:+.4f} to {high:+.4f}   width {width:.4f}")
        if len(spread) >= 3:
            print(f"  mean {statistics.mean(spread):+.4f}  "
                  f"stdev {statistics.stdev(spread):.4f}")
        notes.append(f"spread of final_minus_best over {len(spread)} seeds: "
                     f"{low:+.6f} to {high:+.6f}, width {width:.6f}")
        for ref in (r for r in rows if r["role"] == "reference"):
            value = float(ref["final_minus_best"])
            inside = low <= value <= high
            print(f"  {ref['label']}: {value:+.4f} -- "
                  f"{'inside' if inside else 'OUTSIDE'} that range")
            notes.append(f"reference {ref['label']} at {value:+.6f} is "
                         f"{'inside' if inside else 'outside'} the measured range")

    write_tsv(args.out, COLUMNS, rows, {
        "model": "compounds gpt2 (columns: label,seed,batch) -- runs that differ only "
                 "in the seed, plus any reference run named for comparison",
        "source": "each run's own logging_*.csv beside its checkpoints; seed, batch, "
                  "max_iters and commit from its run_manifest.json",
        "generated": stamp("scripts/compounds_seedvar_spread.py"),
        "note": notes,
        "job": os.environ.get("SLURM_JOB_ID") or None,
    })
    print(f"\n{len(rows)} rows -> {args.out}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
