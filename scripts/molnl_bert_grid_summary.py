#!/usr/bin/env python3
"""Where each BERT run of a molecule_nat_lang grid got to, read from its own state.

Reports the best eval_loss_mask and where it fell, the last value, and how far through
the schedule the run is -- for finished runs and running ones alike, so a progress report
and a final one come from the same place.

With --against, each run is paired with the run of the same size and rate under a
different schedule, which is what says whether doubling the budget bought anything.

    python scripts/molnl_bert_grid_summary.py --runs-root <dir> --match 'bert_*_24k_lr*' \
        --against 'bert_{size}_lr{tag}'
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import re
import statistics as st
import subprocess

NAME = re.compile(r"^bert_(?P<size>small|medium|large|xl)(?:_(?P<schedule>\w+?))?"
                  r"_lr(?P<tag>[0-9p]+e\d)(?:_seed(?P<seed>\d+))?$")
SIZE_ORDER = {"small": 0, "medium": 1, "large": 2, "xl": 3}


def rate_of(tag):
    mantissa, exponent = tag.split("e")
    return float(mantissa.replace("p", ".")) * 10 ** -int(exponent)


def window(points, last_n=10):
    """Mean and spread of the last ``last_n`` evaluations.

    The same rule the GPT-2 grid is judged by, for the same reason: the minimum of a
    series is biased downward by the act of picking it, and a gap smaller than what an
    arm moves inside its own window is not a gap that can be read.
    """
    tail = [v for _, v in points[-last_n:]]
    return (st.mean(tail), st.pstdev(tail) if len(tail) > 1 else 0.0)


def run_state(run_dir):
    """What Slurm says about the job that wrote this directory, or None.

    Steps short of max_steps means "still going" only while the job is alive. A run that
    was cancelled leaves exactly the same directory as one mid-flight, and calling that
    "running" in a report is how a stopped arm gets waited for.
    """
    log = os.path.join(run_dir, "segments.log")
    if not os.path.exists(log):
        return None
    job = None
    for line in open(log):
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            job = parts[1]
    if job is None:
        return None
    try:
        out = subprocess.run(["sacct", "-X", "-n", "-j", job, "-o", "State%20"],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.split()[0] if out.split() else None


def read_run(run_dir):
    """(points, max_steps) from the newest checkpoint's trainer_state.json."""
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[1]))
    if not cks:
        return [], None
    state = json.load(open(os.path.join(cks[-1], "trainer_state.json")))
    pts = sorted((e["step"], e["eval_loss_mask"]) for e in state["log_history"]
                 if "eval_loss_mask" in e)
    return pts, int(state.get("max_steps") or 0)


def collect(runs_root, pattern):
    runs = {}
    for d in sorted(glob.glob(os.path.join(runs_root, "bert_*"))):
        name = os.path.basename(d)
        if not fnmatch.fnmatch(name, pattern):
            continue
        m = NAME.match(name)
        if not m:
            continue
        pts, max_steps = read_run(d)
        if pts:
            runs[(m["size"], rate_of(m["tag"]), m["tag"], int(m["seed"] or 42))] = {
                "name": name, "points": pts, "max_steps": max_steps,
                "state": run_state(d), "dir": d,
            }
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--match", default="bert_*_lr*")
    ap.add_argument("--against", help="template for the run to compare against, "
                                      "e.g. 'bert_{size}_lr{tag}'")
    a = ap.parse_args()

    runs = collect(a.runs_root, a.match)
    if not runs:
        print(f"nothing matching {a.match} under {a.runs_root}")
        return 1

    print(f"{'run':<26}{'lr':>9}{'step':>8}{'/max':>8}{'window':>9}{'sd':>8}{'best':>9}"
          f"{'at':>8}{'last':>9}  state")
    for key in sorted(runs, key=lambda k: (SIZE_ORDER[k[0]], k[1], k[3])):
        size, lr, tag, seed = key
        run = runs[key]
        pts = run["points"]
        reached = pts[-1][0]
        best_step, best = min(pts, key=lambda p: p[1])
        done = run["max_steps"] and reached >= run["max_steps"]
        state = run.get("state")
        if done:
            label = "finished"
        elif state in ("RUNNING", "PENDING", "REQUEUED", "COMPLETING", None):
            label = "running"
        else:
            label = state.split("+")[0].lower()   # CANCELLED by us, TIMEOUT, FAILED
        mean, sd = window(pts)
        print(f"{run['name']:<26}{lr:>9g}{reached:>8,}{run['max_steps']:>8,}{mean:>9.4f}"
              f"{sd:>8.4f}{best:>9.4f}{best_step:>8,}{pts[-1][1]:>9.4f}  "
              f"{label}")

    if a.against:
        other = collect(a.runs_root, "bert_*_lr*")
        print(f"\n{'size':<8}{'lr':>9}{'this':>9}{'other':>9}{'change':>9}  compared with "
              f"(window means)")
        for (size, lr, tag, seed) in sorted(runs, key=lambda k: (SIZE_ORDER[k[0]], k[1], k[3])):
            want = a.against.format(size=size, tag=tag)
            match = [v for k, v in other.items() if v["name"] == want]
            if not match:
                continue
            mine, _ = window(runs[(size, lr, tag, seed)]["points"])
            theirs, _ = window(match[0]["points"])
            print(f"{size:<8}{lr:>9g}{mine:>9.4f}{theirs:>9.4f}{mine - theirs:>+9.4f}  {want}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
