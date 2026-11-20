"""The batch-size grid's loss curves and per-arm summary, read from what the runs wrote.

2026-11-19c §10 asks, for every arm: the lowest validation loss and the column count it
sits at, the value at the budget, the step reached, and whether the arm collapsed. §12
asks for one curve file holding all the arms, with the batch and the columns and tokens
read so far added to the columns the 2026-11-02 curve file carried.

Both come from one pass over what the runs wrote, so the summary cannot disagree with
the curve it is read off. The arm's identity comes from the inventory rather than from
the log, and the log is found by the job name the submitter built out of that same
inventory, so an arm cannot be mislabelled by a header line.

    compounds_batch_sweep_curves.py --arms rt1-...tsv --logs <dir> \
        --curves-out rt1-...tsv --summary-out rt1-...tsv [--arch gpt2]

The GPT-2 reader takes nanoGPT's own evaluation line out of the run's log. The BERT
reader is not here yet: those arms write trainer_state.json and were still running when
this was written, and a reader with no run to read is a reader nobody has checked.

Read-only over the runs: it writes two TSVs and launches nothing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_tsv import check_name, stamp, write_tsv  # noqa: E402

#: nanoGPT's evaluation line.
STEP = re.compile(r"step\s+(\d+):\s+train loss\s+([\d.]+),\s+val loss\s+([\d.]+)")

SEQUENCE_LENGTH = 1024
SEED = 42

#: The degenerate floor measured for this corpus. An arm whose validation loss ends
#: above it has not learned context; it is reported rather than left to be noticed.
DEGENERATE_FLOOR = 2.395

CURVE_COLUMNS = ["arch", "size", "batch", "lr", "seed", "step", "metric", "value",
                 "rows_seen", "tokens_seen", "source"]
SUMMARY_COLUMNS = ["arch", "size", "batch", "lr", "steps_planned", "steps_reached",
                   "eval_points", "judged_metric", "best_value", "best_step",
                   "best_rows", "best_tokens", "final_value", "final_step",
                   "collapsed", "source"]


def read_arms(path, arch):
    with open(path, newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    rows = [r for r in csv.DictReader(lines, delimiter="\t") if r["arch"] == arch]
    if not rows:
        raise SystemExit(f"{path}: no {arch} arm")
    return rows


def log_for(row, log_dir):
    """The arm's log, found by the job name the submitter built from the inventory."""
    stem = (f"mc-cmp-{row['arch']}-{row['size']}-b{row['batch']}"
            f"-lr{row['lr_slug']}")
    found = sorted(glob.glob(os.path.join(log_dir, f"{stem}-*.out")))
    return found


def series(paths):
    """Every evaluation point in these logs, in step order, later wins on a repeat."""
    points = {}
    for path in paths:
        with open(path, errors="ignore") as handle:
            for step, train, val in STEP.findall(handle.read()):
                points[int(step)] = (float(train), float(val))
    return [(s, *points[s]) for s in sorted(points)]


def rows_for(arms, log_dir):
    curves, summary, missing = [], [], []
    for arm in arms:
        paths = log_for(arm, log_dir)
        if not paths:
            missing.append(arm)
            continue
        points = series(paths)
        if not points:
            missing.append(arm)
            continue
        batch = int(arm["batch"])
        source = " ".join(os.path.basename(p) for p in paths)
        for step, train, val in points:
            for metric, value in (("train_loss", train), ("val_loss", val)):
                curves.append({
                    "arch": arm["arch"], "size": arm["size"], "batch": batch,
                    "lr": arm["lr"], "seed": SEED, "step": step,
                    "metric": metric, "value": f"{value:.6f}",
                    "rows_seen": step * batch,
                    "tokens_seen": step * batch * SEQUENCE_LENGTH,
                    "source": source,
                })
        best_val, best_step = min((v, s) for s, _, v in points)
        final_step, _, final_val = points[-1]
        summary.append({
            "arch": arm["arch"], "size": arm["size"], "batch": batch, "lr": arm["lr"],
            "steps_planned": int(arm["steps"]), "steps_reached": final_step,
            "eval_points": len(points),
            "judged_metric": arm["judge_metric"],
            "best_value": f"{best_val:.6f}", "best_step": best_step,
            "best_rows": best_step * batch,
            "best_tokens": best_step * batch * SEQUENCE_LENGTH,
            "final_value": f"{final_val:.6f}", "final_step": final_step,
            "collapsed": "yes" if final_val >= DEGENERATE_FLOOR else "no",
            "source": source,
        })
    return curves, summary, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--logs", required=True, help="the directory holding the run logs")
    ap.add_argument("--arch", default="gpt2", choices=["gpt2"],
                    help="only gpt2 for now; see the module docstring")
    ap.add_argument("--curves-out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args(argv)
    check_name(args.curves_out)
    check_name(args.summary_out)

    arms = read_arms(args.arms, args.arch)
    curves, summary, missing = rows_for(arms, args.logs)
    print(f"=== {len(arms)} {args.arch} arm(s); {len(summary)} read, "
          f"{len(missing)} without a log ===")
    for arm in missing:
        print(f"  ** no evaluation point for {arm['size']} batch {arm['batch']} "
              f"lr {arm['lr']}")

    print(f"\n{'size':8s} {'batch':>5s} {'lr':>9s} {'reached':>8s} {'points':>6s} "
          f"{'best':>9s} {'at step':>8s} {'at tokens':>14s} {'final':>9s}")
    for row in summary:
        print(f"{row['size']:8s} {row['batch']:>5d} {row['lr']:>9s} "
              f"{row['steps_reached']:>8,} {row['eval_points']:>6d} "
              f"{row['best_value']:>9s} {row['best_step']:>8,} "
              f"{row['best_tokens']:>14,} {row['final_value']:>9s}"
              + ("   COLLAPSED" if row["collapsed"] == "yes" else ""))

    print("\n=== lowest batch per size, with the gap to the runner-up ===")
    for size in ("small", "medium", "large", "xl"):
        ranked = sorted((float(r["best_value"]), r["batch"])
                        for r in summary if r["size"] == size)
        if len(ranked) < 2:
            continue
        (first, b1), (second, b2) = ranked[0], ranked[1]
        print(f"  {size:8s} batch {b1:>5d} at {first:.6f}, then batch {b2:>5d} at "
              f"{second:.6f} -- gap {second - first:.6f}")

    shared = {
        "generated": stamp("scripts/compounds_batch_sweep_curves.py"),
        "job": os.environ.get("SLURM_JOB_ID") or None,
    }
    write_tsv(args.curves_out, CURVE_COLUMNS, curves, dict(shared, **{
        "model": f"several (columns: size,batch,lr) -- the compounds batch-size grid's "
                 f"{args.arch} arms",
        "source": "the evaluation line each run wrote in its own log; the arm's "
                  "identity comes from the derived inventory, not from the log",
        "note": [
            "one row per step and metric; rows_seen = step x batch and tokens_seen = "
            f"rows_seen x {SEQUENCE_LENGTH}",
            "the 2026-11-02 curve file's split column is replaced by a metric column: "
            "a value column needs one (column convention R12)",
            f"seed {SEED} on every arm",
        ],
    }))
    write_tsv(args.summary_out, SUMMARY_COLUMNS, summary, dict(shared, **{
        "model": f"several (columns: size,batch,lr) -- one row per {args.arch} arm",
        "source": "read off the same evaluation series as the curve file, in one pass",
        "note": [
            "best_value is the lowest value of the judged metric over the arm's own "
            "evaluation points; best_rows and best_tokens are where it sits",
            "final_value is the value at the budget, which every arm here reached",
            f"collapsed = yes when the final value is at or above {DEGENERATE_FLOOR}, "
            "the degenerate floor measured for this corpus",
        ],
    }))
    print(f"\n{len(curves)} curve rows -> {args.curves_out}")
    print(f"{len(summary)} arm rows -> {args.summary_out}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
