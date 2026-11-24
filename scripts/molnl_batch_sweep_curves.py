#!/usr/bin/env python3
"""The molnl batch-size grid's loss curves and per-arm summary, read from the runs.

The 2026-11-20 directive asks for two things that have to agree with each other:

  §10   per arm, the lowest value of the judged metric and the block count it sits at,
        where that minimum falls as a fraction of the budget, the value at the end of
        the budget and its distance from the minimum, the steps reached, and whether
        the arm collapsed and at which block count. Then, per size, a table over the
        five batches naming the rate that won, with the gap between the first and the
        second so that "batch X is lowest" can be read with its margin.
  §12   one curve file covering all 68 arms, columns aligned with the compounds,
        protein and rna grids, carrying the batch, the blocks read and the tokens.

Both come out of one pass over what the runs wrote, so the summary cannot disagree
with the curve it was read off.

What is read, per architecture:

  bert  the last checkpoint's ``trainer_state.json``. Its ``log_history`` holds three
        kinds of entry per evaluation: the training loss on its own steps, ``eval_loss``
        with its timings, and the position breakdown ``eval_loss_mask`` /
        ``eval_loss_copy`` / ``eval_loss_random``. §7 judges on eval_loss_mask and asks
        for the other three in the file as well, so all of them become rows.
  gpt2  the run's ``logging_*.csv``, which nanoGPT writes one row per evaluation:
        step, train loss, validation loss.

The arm's identity -- its size, batch, rate and budget -- comes from the derived
inventory, never from a log header, so an arm cannot be mislabelled by the thing it is
being measured from. An arm whose output cannot be read is named in the output and
makes the exit code non-zero rather than being left out quietly.

Collapse. Two tests, both measurable on the arm itself:

  a non-finite value at any point, and
  a value, after the arm's minimum, at or above its own first evaluation point -- that
      is, the run has come all the way back to where it had not learned anything.

A floor can be given with --floor <value> --floor-source <where it was measured>, and
then a point at or above it (after the minimum) counts too. There is no default: the
numbers on record for this corpus are per split, per metric and per architecture, and
a floor quoted from the wrong one of those would decide this column wrongly. The two
tests above need no constant.

    molnl_batch_sweep_curves.py --arms rt1-...tsv --root <MODEL_OUTPUT_ROOT> \
        --curves-out rt1-...tsv --summary-out rt1-...tsv [--arch bert|gpt2|both] \
        [--band 0.0015] [--floor 4.6449 --floor-source "..."]

Read-only over the runs: it writes two TSVs and launches nothing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_tsv import check_name, stamp, write_tsv  # noqa: E402

SEQUENCE_LENGTH = 1024
SEED = 42

#: Every metric that becomes rows, and which side of the split it belongs to.
BERT_METRICS = (
    ("loss", "train", "train_loss_mask"),
    ("eval_loss_mask", "valid", "eval_loss_mask"),
    ("eval_loss", "valid", "eval_loss"),
    ("eval_loss_copy", "valid", "eval_loss_copy"),
    ("eval_loss_random", "valid", "eval_loss_random"),
)

CURVE_COLUMNS = ["arch", "size", "batch", "lr", "seed", "step", "split", "metric",
                 "value", "rows_seen", "tokens_seen", "source"]
SUMMARY_COLUMNS = ["arch", "size", "batch", "lr", "lr_multiplier", "judged_metric",
                   "steps_planned", "steps_reached", "eval_points",
                   "best_value", "best_step", "best_rows", "best_tokens",
                   "best_fraction_of_budget", "final_value", "final_step",
                   "final_minus_best", "collapsed", "collapse_step", "collapse_rows",
                   "source"]


def read_arms(path, arch):
    with open(path, newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    if arch != "both":
        rows = [r for r in rows if r["arch"] == arch]
    if not rows:
        raise SystemExit(f"{path}: no arm to read")
    return rows


def run_dir(root, arm):
    base = "bert-output" if arm["arch"] == "bert" else "gpt2-output"
    return os.path.join(root, "molecule_nat_lang", base, arm["out_dir"])


def bert_series(path):
    """Every logged value of this run, as {step: {metric: value}}, plus the source."""
    cks = sorted(glob.glob(os.path.join(path, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[1]))
    if not cks:
        return {}, ""
    state_path = os.path.join(cks[-1], "trainer_state.json")
    if not os.path.exists(state_path):
        return {}, ""
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    points: dict[int, dict[str, float]] = {}
    for entry in state.get("log_history", []):
        step = entry.get("step")
        if step is None:
            continue
        for key, _split, name in BERT_METRICS:
            if key in entry:
                points.setdefault(int(step), {})[name] = float(entry[key])
    source = os.path.join(os.path.basename(path), os.path.basename(cks[-1]),
                          "trainer_state.json")
    return points, source


def gpt2_series(path):
    paths = sorted(glob.glob(os.path.join(path, "logging_*.csv")))
    points: dict[int, dict[str, float]] = {}
    for one in paths:
        with open(one, encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if not row or not row[0].strip().isdigit():
                    continue
                points[int(row[0])] = {"train_loss": float(row[1]),
                                       "val_loss": float(row[2])}
    source = " ".join(os.path.basename(p) for p in paths)
    return points, (f"{os.path.basename(path)}/{source}" if source else "")


def judged(points, metric):
    """The judged metric's series as [(step, value)], in step order."""
    return [(step, values[metric]) for step, values in sorted(points.items())
            if metric in values]


def collapse_at(series, floor):
    """The first step at which this arm counts as collapsed, or None.

    A non-finite value counts wherever it is. The other two tests only apply after the
    minimum: before it the run has not descended yet, and its own first evaluation
    point is above both thresholds by construction.
    """
    for step, value in series:
        if not math.isfinite(value):
            return step
    if len(series) < 2:
        return None
    start = series[0][1]
    best_index = min(range(len(series)), key=lambda i: series[i][1])
    for step, value in series[best_index + 1:]:
        if value >= start or (floor is not None and value >= floor):
            return step
    return None


def rows_for(arms, root, floor):
    curves, summary, missing = [], [], []
    for arm in arms:
        path = run_dir(root, arm)
        reader = bert_series if arm["arch"] == "bert" else gpt2_series
        points, source = reader(path)
        metric = arm["judge_metric"]
        series = judged(points, metric)
        if not series:
            missing.append((arm, path))
            continue
        batch = int(arm["batch"])
        planned = int(arm["steps"])
        for step, values in sorted(points.items()):
            for name, value in sorted(values.items()):
                split = "train" if name.startswith("train") else "valid"
                curves.append({
                    "arch": arm["arch"], "size": arm["size"], "batch": batch,
                    "lr": arm["lr"], "seed": SEED, "step": step,
                    "split": split, "metric": name, "value": f"{value:.6f}",
                    "rows_seen": step * batch,
                    "tokens_seen": step * batch * SEQUENCE_LENGTH,
                    "source": source,
                })
        best_value, best_step = min((v, s) for s, v in series)
        final_step, final_value = series[-1]
        hit = collapse_at(series, floor)
        summary.append({
            "arch": arm["arch"], "size": arm["size"], "batch": batch, "lr": arm["lr"],
            "lr_multiplier": arm["lr_multiplier"], "judged_metric": metric,
            "steps_planned": planned, "steps_reached": final_step,
            "eval_points": len(series),
            "best_value": f"{best_value:.6f}", "best_step": best_step,
            "best_rows": best_step * batch,
            "best_tokens": best_step * batch * SEQUENCE_LENGTH,
            "best_fraction_of_budget": f"{best_step / planned:.4f}",
            "final_value": f"{final_value:.6f}", "final_step": final_step,
            "final_minus_best": f"{final_value - best_value:+.6f}",
            "collapsed": "yes" if hit is not None else "no",
            "collapse_step": hit if hit is not None else "",
            "collapse_rows": hit * batch if hit is not None else "",
            "source": source,
        })
    return curves, summary, missing


def print_arms(summary):
    print(f"\n{'arch':5s} {'size':8s} {'batch':>5s} {'lr':>9s} {'reached':>9s} "
          f"{'points':>6s} {'best':>9s} {'at step':>9s} {'of budget':>9s} "
          f"{'final':>9s} {'final-best':>10s}")
    for row in summary:
        print(f"{row['arch']:5s} {row['size']:8s} {row['batch']:>5d} {row['lr']:>9s} "
              f"{row['steps_reached']:>9,} {row['eval_points']:>6d} "
              f"{row['best_value']:>9s} {row['best_step']:>9,} "
              f"{float(row['best_fraction_of_budget']) * 100:>8.1f}% "
              f"{row['final_value']:>9s} {row['final_minus_best']:>10s}"
              + (f"   COLLAPSED at {row['collapse_step']}"
                 if row["collapsed"] == "yes" else ""))


def print_tables(summary, band):
    """§10's per-size tables: which rate won each batch, and which batch won the size."""
    for arch in ("bert", "gpt2"):
        rows = [r for r in summary if r["arch"] == arch]
        if not rows:
            continue
        for size in ("small", "medium", "large", "xl"):
            here = [r for r in rows if r["size"] == size]
            if not here:
                continue
            print(f"\n=== {arch} {size}: the batches, and the rate that won each ===")
            winners = []
            for batch in sorted({r["batch"] for r in here}):
                cell = sorted((float(r["best_value"]), r) for r in here
                              if r["batch"] == batch)
                value, row = cell[0]
                tried = sorted(float(r["lr_multiplier"]) for r in here
                               if r["batch"] == batch)
                mult = float(row["lr_multiplier"])
                edge = ("  ** the winning rate is at the edge of the multipliers tried "
                        f"({', '.join(f'{m:g}' for m in tried)})"
                        if len(tried) > 1 and mult in (tried[0], tried[-1]) else "")
                print(f"  batch {batch:>5d}  lr {row['lr']:>9s} "
                      f"(x{mult:g})  {value:.6f}{edge}")
                winners.append((value, batch, row["lr"]))
            winners.sort()
            if len(winners) >= 2:
                (first, b1, lr1), (second, b2, _) = winners[0], winners[1]
                gap = second - first
                print(f"  lowest: batch {b1} at lr {lr1}, {first:.6f}; then batch {b2} "
                      f"at {second:.6f} -- gap {gap:.6f}")
                if band is None:
                    print("  (no reproducibility band was given, so this is a ranking "
                          "and not a claim that batch "
                          f"{b1} is the minimum)")
                elif gap < band:
                    print(f"  ** the gap is inside the {band:g} reproducibility band: "
                          "the ranking is not a minimum")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--root", required=True,
                    help="the grid's MODEL_OUTPUT_ROOT; the run directories are under "
                         "<root>/molecule_nat_lang/{bert,gpt2}-output/")
    ap.add_argument("--arch", default="both", choices=["bert", "gpt2", "both"])
    ap.add_argument("--curves-out", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--band", type=float, default=None,
                    help="the reproducibility band to read the first-to-second gap "
                         "against; without it no minimum is claimed")
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--floor-source", default=None,
                    help="where --floor was measured; required with --floor")
    args = ap.parse_args(argv)
    check_name(args.curves_out)
    check_name(args.summary_out)
    if (args.floor is None) != (args.floor_source is None):
        raise SystemExit("--floor and --floor-source go together: a number whose "
                         "provenance is not in the file cannot be checked later")

    arms = read_arms(args.arms, args.arch)
    curves, summary, missing = rows_for(arms, args.root, args.floor)
    print(f"=== {len(arms)} arm(s) in the inventory; {len(summary)} read, "
          f"{len(missing)} without a readable series ===")
    for arm, path in missing:
        print(f"  ** nothing to read for {arm['arch']} {arm['size']} batch "
              f"{arm['batch']} lr {arm['lr']}: {path}")
    if summary:
        print_arms(summary)
        print_tables(summary, args.band)

    floor_note = ("a point at or above the floor "
                  f"{args.floor:g} ({args.floor_source}), after the minimum"
                  if args.floor is not None else
                  "no floor was given, so only the two tests above were applied")
    shared = {
        "generated": stamp("scripts/molnl_batch_sweep_curves.py"),
        "job": os.environ.get("SLURM_JOB_ID") or None,
    }
    write_tsv(args.curves_out, CURVE_COLUMNS, curves, dict(shared, **{
        "model": "several (columns: arch,size,batch,lr) -- the molnl batch-size grid",
        "source": "what each run wrote: an HF run's trainer_state.json at its last "
                  "checkpoint, a nanoGPT run's logging_*.csv. The arm's identity comes "
                  "from the derived inventory, not from the log",
        "note": [
            "one row per step and metric; rows_seen = step x batch and tokens_seen = "
            f"rows_seen x {SEQUENCE_LENGTH}. A row is one 1,024-token block, which is "
            "the unit the 2026-11-20 directive fixes the budget in",
            "the columns are the compounds grid's, with the split column the molnl "
            "2026-11-12 curve file carried kept alongside the metric column",
            "BERT contributes train_loss_mask, eval_loss_mask (the judged metric, §7), "
            "eval_loss, eval_loss_copy and eval_loss_random; GPT-2 contributes "
            "train_loss and val_loss",
            f"seed {SEED} on every arm",
        ],
    }))
    write_tsv(args.summary_out, SUMMARY_COLUMNS, summary, dict(shared, **{
        "model": "several (columns: arch,size,batch,lr) -- one row per arm",
        "source": "read off the same series as the curve file, in one pass",
        "note": [
            "best_value is the lowest value of the arm's judged metric over its own "
            "evaluation points; best_rows and best_tokens are where it sits",
            "best_fraction_of_budget = best_step / steps_planned, and "
            "final_minus_best = final_value - best_value (§10)",
            "collapsed = yes on a non-finite value anywhere, or on a value after the "
            "minimum at or above the arm's own first evaluation point; " + floor_note,
            "collapse_step and collapse_rows are the first such point",
        ],
    }))
    print(f"\n{len(curves)} curve rows -> {args.curves_out}")
    print(f"{len(summary)} arm rows -> {args.summary_out}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
