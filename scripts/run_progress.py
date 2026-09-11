#!/usr/bin/env python3
"""Progress of HF Trainer jobs, from the step counter, the clock and Slurm.

tqdm's "s/it" is a smoothed recent rate, not an average. On the mol_nl BERT runs it
read 39.98 while the average was 35.04 and the last hundred steps ran at 40.00. Two
estimates went to the boss built on that number (27.19 in August, 34.29 / 58.62 in
September), and a segment's end date was once derived from a "now" typed by hand.

So this prints nothing tqdm computed. Rates are elapsed seconds over steps, taken
from the progress lines; times come from sacct and squeue; "now" is the system
clock. Numbers in a report should be copied from here, not read off a log.

A resumed segment restarts tqdm's clock at the step it resumed from, so every rate is
for this segment only -- from the first progress line this job printed to its last.

    python scripts/run_progress.py 87693 87694 [--save-every 250] [--window 500]
"""

import argparse
import datetime as dt
import glob
import os
import re
import subprocess

# tqdm writes elapsed as MM:SS under an hour and H:MM:SS from then on (hours keep
# counting past 24). Reading only the second form drops the first hour of every run,
# which for a segment of an hour or two is most of it.
# A resumed segment prints a 0/max_steps line before tqdm jumps to the resumed step,
# with no time in between. Taken as the start, it divides this segment's time by steps
# earlier segments ran -- 87484 read 27.75 s/it for a run at 58. HF says where it
# resumed from, so the segment starts there.
RESUMED = re.compile(r"Resuming training from: .*checkpoint-(\d+)")
PROGRESS = re.compile(r"\|\s*(\d+)/(\d+) \[(?:(\d+):)?(\d\d):(\d\d)<")


def _repo_root():
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], capture_output=True,
                                text=True, check=True).stdout.strip()
        return os.path.dirname(os.path.abspath(common))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.getcwd()


def _slurm(job):
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", str(job), "-o",
                          "JobName,State,NodeList,Start,End,Timelimit"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return None
    name, state, node, start, end, limit = out[0].split("|")
    if state.split()[0] in ("RUNNING", "PENDING"):
        end = subprocess.run(["squeue", "-h", "-j", str(job), "-o", "%e"],
                             capture_output=True, text=True).stdout.strip() or end
    return {"name": name, "state": state.split()[0], "node": node, "start": start,
            "end": end, "limit": limit}


def _progress(path):
    """Training-bar steps -> elapsed seconds, for this segment only.

    HF prints evaluation bars in the same "N/M [elapsed<" form. Their M is the number of
    eval batches, and their N would overwrite training steps of the same number, so only
    the bar with the largest total -- max_steps -- is kept. Steps before the checkpoint
    this segment resumed from are dropped (see RESUMED).
    """
    by_total = {}
    with open(path, errors="ignore") as fh:
        text = fh.read()
    for m in PROGRESS.finditer(text):
        h = int(m.group(3) or 0)
        by_total.setdefault(int(m.group(2)), {})[int(m.group(1))] = (
            h * 3600 + int(m.group(4)) * 60 + int(m.group(5)))
    if not by_total:
        return {}, None, 0
    total = max(by_total)
    r = RESUMED.search(text)
    start = int(r.group(1)) if r else 0
    return {k: v for k, v in by_total[total].items() if k >= start}, total, start


def _rate(steps, a, b):
    """Seconds per step between the last progress lines at or before steps a and b."""
    ks = sorted(steps)
    ka = [k for k in ks if k <= a] or ks[:1]
    kb = [k for k in ks if k <= b] or ks[:1]
    ka, kb = ka[-1], kb[-1]
    return (steps[kb] - steps[ka]) / (kb - ka) if kb > ka else None


def report(job, save_every, window, root):
    s = _slurm(job)
    logs = sorted(glob.glob(os.path.join(root, "workflows", "slurm-logs", f"*-{job}.out")))
    print(f"=== {job}  {s['name'] if s else '?'}  {s['state'] if s else 'sacct: not found'}"
          f"{'  on ' + s['node'] if s else ''} ===")
    if s:
        print(f"  start {s['start']}   end {s['end']}   limit {s['limit']}   (sacct/squeue)")
    if not logs:
        print("  log: not found under workflows/slurm-logs")
        return
    steps, total, resumed = _progress(logs[0])
    if len(steps) < 2:
        print(f"  log: {os.path.basename(logs[0])} has no HF progress lines")
        return
    ks = sorted(steps)
    first, last = ks[0], ks[-1]
    avg = (steps[last] - steps[first]) / (last - first)
    recent = _rate(steps, last - 100, last)
    print(f"  this segment: step {first:,} -> {last:,} of {total:,}"
          f"{' (resumed)' if resumed else ''}   "
          f"{(steps[last] - steps[first]) / 3600:.1f} h   mean {avg:.2f} s/it   "
          f"last 100 steps {recent:.2f} s/it" if recent else "")
    lows = list(range(first - first % window, last - window + 1, window))
    cells = [f"{lo:,}-{lo + window:,}: {_rate(steps, lo, lo + window):.1f}"
             for lo in lows[-8:] if _rate(steps, lo, lo + window)]
    if cells:
        print(f"  per {window} steps (s/it): " + "   ".join(cells))
    if s and s["state"] == "RUNNING" and s["end"] not in ("", "Unknown"):
        left = (dt.datetime.fromisoformat(s["end"]) - dt.datetime.now()).total_seconds()
        at_end = min(total, last + max(left, 0) / avg)
        print(f"  at the time limit ({s['end']}): step ~{at_end:,.0f} at the mean rate"
              f" -> resume from checkpoint-{int(at_end // save_every * save_every):,}"
              f"   ({left / 3600:.1f} h left, now {dt.datetime.now():%Y-%m-%d %H:%M})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--window", type=int, default=500)
    a = ap.parse_args()
    root = _repo_root()
    for job in a.jobs:
        report(job, a.save_every, a.window, root)


if __name__ == "__main__":
    main()
