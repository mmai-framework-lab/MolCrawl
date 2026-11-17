#!/usr/bin/env python3
"""Where each mol_nl extension run of the 2026-11-16 order stands, read from the runs.

One row per run: how far it is, what it has improved on since the step it resumed from,
how many evaluations it has gone without a new best, and where early stopping would fire
if it goes without one from here.

Every number is read from what the run wrote -- an HF run's trainer_state.json at its
newest checkpoint, a nanoGPT run's logging_*.csv -- except the projected stopping step,
which is arithmetic on the patience counter and is labelled as a projection. The
value a run resumed from is passed in (``--baseline``) rather than re-derived, because
for BERT large that value sits at step 23,800 in the run this one continues, not at the
last step of it.

    python scripts/molnl_extend_status.py --bert-root <dir> --gpt2-root <dir> [--markdown]

Patience and the evaluation interval are read from each run as well: BERT from the
callback state HF stores in trainer_state.json, GPT-2 from the gap between its logged
iterations, so a config change cannot leave this script quoting the old cadence.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess

#: run slug -> (arch, size, job, the value and step it resumed from, the constant rate)
RUNS = (
    ("bert_small_24k_lr1e3_ext", "bert", "small", 356041, 0.059050, 24000, 1e-4),
    ("bert_medium_24k_lr3e4_ext", "bert", "medium", 356043, 0.056791, 24000, 3e-5),
    ("bert_large_24k_lr1p7e4_ext", "bert", "large", 356045, 0.057246, 23800, 1.7e-5),
    ("gpt2_small_1500_lr1p2e3_ext", "gpt2", "small", 355773, 0.5795, 1450, 1.2e-4),
    ("gpt2_medium_1500_lr6e4_ext", "gpt2", "medium", 355775, 0.5682, 1450, 6e-5),
    ("gpt2_large_1500_lr3e4_ext", "gpt2", "large", 355777, 0.5536, 1450, 3e-5),
    ("gpt2_xl_1500_lr3e4_ext", "gpt2", "xl", 355779, 0.5645, 1450, 3e-5),
)
RESUME_STEP = {"bert": 24000, "gpt2": 1500}
#: hours per 1,000 steps, measured on the runs being continued (4 GPUs each)
HOURS_PER_1K = {("bert", "small"): 1.11, ("bert", "medium"): 2.11, ("bert", "large"): 3.33,
                ("gpt2", "small"): 0.71, ("gpt2", "medium"): 1.28, ("gpt2", "large"): 2.27,
                ("gpt2", "xl"): 5.03}


def slurm(job):
    out = subprocess.run(["sacct", "-X", "-n", "-j", str(job), "-o", "State,Elapsed,Start"],
                         capture_output=True, text=True).stdout.split()
    return (out + ["?", "?", "?"])[:3]


def bert(run_dir):
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[1]))
    state = json.load(open(os.path.join(cks[-1], "trainer_state.json"), encoding="utf-8"))
    series = sorted((x["step"], x["eval_loss_mask"]) for x in state["log_history"]
                    if "eval_loss_mask" in x)
    callback = state.get("stateful_callbacks", {}).get("LoggedMetricEarlyStopping", {})
    interval = min((b[0] - a[0] for a, b in zip(series, series[1:])), default=100)
    best = str(state.get("best_model_checkpoint") or "")
    return {
        "step": state["global_step"], "ceiling": state["max_steps"], "series": series,
        "best_value": state.get("best_metric"),
        "best_step": int(best.rsplit("-", 1)[1]) if best else None,
        "counter": callback.get("attributes", {}).get("early_stopping_patience_counter"),
        "patience": callback.get("args", {}).get("early_stopping_patience"),
        "interval": interval,
        "checkpoints": [int(os.path.basename(c).split("-")[1]) for c in cks],
        "metric": "eval_loss_mask",
    }


def gpt2(run_dir):
    series = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "logging_*.csv"))):
        for row in csv.DictReader(open(path, encoding="utf-8"), skipinitialspace=True):
            series[int(row["iter"])] = float(row["val_loss"])
    rows = sorted(series.items())
    best = min(rows, key=lambda kv: kv[1])
    interval = min((b[0] - a[0] for a, b in zip(rows, rows[1:])), default=50)
    marker = os.path.join(run_dir, "EARLY_STOPPED")
    cks = sorted(int(os.path.basename(d).split("-")[1])
                 for d in glob.glob(os.path.join(run_dir, "checkpoint-*")))
    return {
        "step": max(series), "ceiling": None, "series": rows,
        "best_value": best[1], "best_step": best[0], "counter": None, "patience": None,
        "interval": interval, "checkpoints": cks, "metric": "val_loss",
        "early_stopped": open(marker, encoding="utf-8").read().strip() if os.path.exists(marker) else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert-root", required=True)
    ap.add_argument("--gpt2-root", required=True)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    out = []
    for slug, arch, size, job, baseline, baseline_step, lr in RUNS:
        root = args.bert_root if arch == "bert" else args.gpt2_root
        run_dir = os.path.join(root, slug)
        info = bert(run_dir) if arch == "bert" else gpt2(run_dir)
        resumed = RESUME_STEP[arch]
        after = [(s, v) for s, v in info["series"] if s > resumed]
        best_after = min(after, key=lambda t: t[1]) if after else None
        state, elapsed, start = slurm(job)
        projected = None
        if info["counter"] is not None and info["patience"]:
            projected = info["step"] + (info["patience"] - info["counter"]) * info["interval"]
        row = dict(arch=arch, size=size, run=slug, job=job, state=state, elapsed=elapsed,
                   start=start, step=info["step"], ceiling=info["ceiling"],
                   metric=info["metric"], constant_lr=lr,
                   baseline=baseline, baseline_step=baseline_step,
                   best_after_resume=best_after[1] if best_after else None,
                   best_after_resume_step=best_after[0] if best_after else None,
                   improved=(best_after[1] < baseline) if best_after else None,
                   adopted_value=info["best_value"], adopted_step=info["best_step"],
                   counter=info["counter"], patience=info["patience"],
                   interval=info["interval"], evals_after_resume=len(after),
                   latest=after[-1] if after else None,
                   projected_stop=projected,
                   hours_left=(round((projected - info["step"]) / 1000 * HOURS_PER_1K[(arch, size)], 1)
                               if projected else None),
                   checkpoints=info["checkpoints"],
                   early_stopped=info.get("early_stopped", ""))
        out.append(row)

    if not args.markdown:
        print(json.dumps(out, indent=2))
        return 0
    for arch, title in (("gpt2", "GPT-2"), ("bert", "BERT")):
        print(f"\n### {title}\n")
        print("| size | job | state | step | resumed from | best after the resume | adopted | "
              "no-improve | projected stop |")
        print("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for r in (x for x in out if x["arch"] == arch):
            print("| {size} | {job} | {state} | {step} | {baseline:.6f} @{baseline_step} | "
                  "{b} @{bs} | {adopted_value:.6f} @{adopted_step} | {c} | {p} |".format(
                      b=f"{r['best_after_resume']:.6f}", bs=r["best_after_resume_step"],
                      c=(f"{r['counter']}/{r['patience']}" if r["counter"] is not None else "-"),
                      p=r["projected_stop"] or "-", **r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
