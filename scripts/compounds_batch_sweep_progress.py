"""Where each arm of the compounds batch-size grid has got to, read off what it wrote.

A progress report wants one line per arm: the step reached against the budget, the
epoch that step is, the best value of the judged metric so far, how many checkpoints
the retention policy is holding, and when the segment now running expects to finish.
All of it comes from the arm's own log and output directory, and the arm's identity
comes from the inventory TSV rather than from a log header, so an arm cannot be
mislabelled.

    compounds_batch_sweep_progress.py --arms rt1-...tsv --logs <dir> \
        --runs <bert output dir> --out rt1-...tsv [--arch bert]

The step and the remaining time are the training loop's own progress line; the
remaining time is that loop's extrapolation inside the segment now running and does
not include the wait for the segments queued behind it.

Read-only over the runs: it writes one TSV and launches nothing.
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

#: The training loop's progress line: step, planned steps, elapsed, remaining.
BAR = re.compile(r"(\d+)/(\d+) \[(\d+:\d+:\d+|\d+:\d+)<(\d+:\d+:\d+|\d+:\d+|\?),")
#: The save-on-improve line, which carries both the value and the step it is at.
IMPROVED = re.compile(r"improved to ([\d.]+) at step (\d+)")
#: The evaluation dictionary the BERT path prints.
EPOCH = re.compile(r"'epoch': ([\d.]+)")

COLUMNS = ["arch", "size", "batch", "lr", "arm", "steps_planned", "steps_reached",
           "fraction_of_budget", "epoch", "judged_metric", "best_value", "best_step",
           "checkpoints_held", "newest_checkpoint", "segment_elapsed_h",
           "segment_remaining_h", "source"]


def hours(text):
    """Turn the loop's h:mm:ss (or mm:ss) into hours; an unknown stays empty."""
    if text == "?":
        return ""
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return f"{parts[0] + parts[1] / 60 + parts[2] / 3600:.2f}"


def read_arms(path, arch):
    with open(path, newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    rows = [r for r in csv.DictReader(lines, delimiter="\t") if r["arch"] == arch]
    if not rows:
        raise SystemExit(f"{path}: no {arch} arm")
    return rows


def newest_log(logs, arm):
    """The newest segment log of this arm. The job name is the arm's name plus -sN."""
    found = sorted(glob.glob(os.path.join(logs, f"{arm}-s*.out")),
                   key=os.path.getmtime)
    return found[-1] if found else None


def checkpoints(run_dir):
    steps = []
    for path in glob.glob(os.path.join(run_dir, "checkpoint-*")):
        match = re.search(r"checkpoint-(\d+)$", path)
        if match:
            steps.append(int(match.group(1)))
    return steps


def job_stem(arm):
    """The job name the submitter built, without the segment suffix.

    The inventory does not carry the job name: the submitter builds it out of the
    architecture, the size, the batch and the rate slug, and the same four columns
    are here. Building it the same way is what ties a log to an arm; the output
    directory cannot do it, because for GPT-2 the directory says ex-large where the
    job name says xl.
    """
    return f"mc-cmp-{arm['arch']}-{arm['size']}-b{arm['batch']}-lr{arm['lr_slug']}"


def row_for(arm, logs, runs):
    name = job_stem(arm)
    log = newest_log(logs, name)
    run_dir = os.path.join(runs, arm["out_dir"])
    held = checkpoints(run_dir) if os.path.isdir(run_dir) else []
    row = {
        "arch": arm["arch"], "size": arm["size"], "batch": arm["batch"],
        "lr": arm["lr"], "arm": name,
        "steps_planned": arm["steps"], "steps_reached": "", "fraction_of_budget": "",
        "epoch": "", "judged_metric": arm.get("judge_metric", ""),
        "best_value": "", "best_step": "",
        "checkpoints_held": str(len(held)),
        "newest_checkpoint": str(max(held)) if held else "",
        "segment_elapsed_h": "", "segment_remaining_h": "",
        "source": os.path.basename(log) if log else "",
    }
    if not log:
        return row, f"no log for {name}"
    with open(log, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    # The log carries two kinds of progress line: the training loop's, whose total is
    # the arm's budget, and one per evaluation, whose total is the number of
    # evaluation batches. Reading the newest line of either kind would report an
    # evaluation's 63/79 as the arm's position, so only the training line counts.
    bars = [b for b in BAR.findall(text) if b[1] == arm["steps"]]
    others = [b for b in BAR.findall(text) if b[1] != arm["steps"]]
    if bars:
        step, _planned, elapsed, remaining = bars[-1]
        row["steps_reached"] = step
        row["fraction_of_budget"] = f"{int(step) / int(arm['steps']):.4f}"
        row["segment_elapsed_h"] = hours(elapsed)
        row["segment_remaining_h"] = hours(remaining)
    elif others:
        return row, (f"{name}: no progress line over {arm['steps']} steps; the log's "
                     f"lines count to {sorted({b[1] for b in others})}")
    epochs = EPOCH.findall(text)
    if epochs:
        row["epoch"] = epochs[-1]
    improved = IMPROVED.findall(text)
    if improved:
        row["best_value"], row["best_step"] = improved[-1]
    return row, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arch", default="bert")
    args = parser.parse_args(argv)

    check_name(args.out)
    arms = read_arms(args.arms, args.arch)
    rows, notes = [], []
    for arm in arms:
        row, note = row_for(arm, args.logs, args.runs)
        rows.append(row)
        if note:
            notes.append(note)

    rows.sort(key=lambda r: (r["size"], int(r["batch"]), r["lr"]))
    print(f"=== {len(rows)} {args.arch} arm(s); "
          f"{sum(1 for r in rows if r['steps_reached'])} with a progress line ===")
    for note in notes:
        print(f"  ** {note}")
    print(f"\n{'arm':38s} {'reached':>9s} {'of budget':>10s} {'epoch':>7s} "
          f"{'best':>9s} {'at step':>9s} {'held':>5s} {'elapsed':>8s} {'left':>8s}")
    for r in rows:
        frac = f"{float(r['fraction_of_budget']) * 100:.2f}%" if r["fraction_of_budget"] else "-"
        print(f"{r['arm']:38s} {r['steps_reached']:>9s} {frac:>10s} {r['epoch']:>7s} "
              f"{r['best_value']:>9s} {r['best_step']:>9s} {r['checkpoints_held']:>5s} "
              f"{r['segment_elapsed_h']:>8s} {r['segment_remaining_h']:>8s}")

    write_tsv(args.out, COLUMNS, rows, {
        "model": f"several (columns: size,batch,lr) -- one row per {args.arch} arm of "
                 "the compounds batch-size grid, in flight",
        "source": "each arm's own segment log and output directory; the arm's "
                  "identity comes from the derived inventory, not from the log",
        "generated": stamp("scripts/compounds_batch_sweep_progress.py"),
        "job": os.environ.get("SLURM_JOB_ID") or None,
        "note": [
            "steps_reached is the training loop's own progress line in the segment "
            "now running, so it is ahead of newest_checkpoint",
            "segment_remaining_h is that loop's extrapolation at its current rate "
            "inside this segment; it excludes the queue wait of the segments behind it",
            "best_value is the last save-on-improve line, so it is the lowest value "
            "of judged_metric the arm has reached, and best_step is where it sits",
            "checkpoints_held is what the retention policy is keeping now (best 10 "
            "by the judged metric plus the newest), not how many were written",
        ],
    })
    print(f"\n{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
