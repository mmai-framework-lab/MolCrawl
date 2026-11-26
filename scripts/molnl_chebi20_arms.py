#!/usr/bin/env python3
"""The eight molnl pretraining runs the ChEBI-20 retrieval order starts from.

The 2026-11-26 order, item 1: per size and architecture, the run whose judged metric is
lowest -- eval_loss_mask for BERT, val_loss for GPT-2 -- whatever its batch and rate.
The runs it chooses among are the 68 arms of the 2026-11-20 batch-size grid, the arm set
the grid's own summary (--grid-summary) lists. That summary is only used to name the
arms; every value written here is read again from what the run itself wrote:

  bert  the last checkpoint's trainer_state.json (log_history), and run_manifest.json
  gpt2  logging_*.csv, checkpoint-<step>/training_args.json, and run_manifest.json

and the chosen checkpoint is checked to be the one the minimum belongs to: for BERT,
trainer_state's best_model_checkpoint and best_metric; for GPT-2, the checkpoint
directory's own iteration and val_loss. A disagreement stops the script.

Item 7 also asks for the eight runs' pretraining curves, so they are written from the
same pass: every evaluation point, the training loss at that step, and for BERT both
eval_loss and eval_loss_mask.

    python scripts/molnl_chebi20_arms.py --runs <.../molecule_nat_lang> \
        --grid-summary <rt1-molnl-bsweep-arm-summary-*.tsv> \
        --arms-out <tsv> --curves-out <tsv>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

SIZES = ("small", "medium", "large", "xl")


def read_tsv(path):
    with open(path, encoding="utf-8") as fh:
        lines = [line for line in fh if not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def last_checkpoint(run_dir):
    found = glob.glob(os.path.join(run_dir, "checkpoint-*"))
    if not found:
        raise SystemExit(f"{run_dir}: no checkpoint")
    return max(found, key=lambda p: int(p.rsplit("-", 1)[1]))


def bert_points(run_dir):
    """{step: {train_loss, eval_loss, eval_loss_mask}} from the last trainer_state."""
    state = json.load(open(os.path.join(last_checkpoint(run_dir), "trainer_state.json")))
    pts = {}
    for e in state["log_history"]:
        p = pts.setdefault(int(e["step"]), {})
        if "loss" in e:
            p["train_loss"] = float(e["loss"])
        if "eval_loss" in e:
            p["eval_loss"] = float(e["eval_loss"])
        if "eval_loss_mask" in e:
            p["eval_loss_mask"] = float(e["eval_loss_mask"])
    return pts, state


def gpt2_points(run_dir):
    """{step: {train_loss, val_loss}} from the run's logging CSVs (one row per evaluation).

    A run can hold more than one: small b512 has a second file with only the header,
    started a second after the first. All are read; a step written twice must carry the
    same values or the script stops.
    """
    logs = sorted(glob.glob(os.path.join(run_dir, "logging_*.csv")))
    if not logs:
        raise SystemExit(f"{run_dir}: no logging CSV")
    pts = {}
    for path in logs:
        with open(path, encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = [h.strip() for h in next(reader)]
            if header != ["iter", "train_loss", "val_loss"]:
                raise SystemExit(f"{path}: unexpected header {header}")
            for row in reader:
                step, tr, va = (x.strip() for x in row)
                val = {"train_loss": float(tr), "val_loss": float(va)}
                if int(step) in pts and pts[int(step)] != val:
                    raise SystemExit(f"{run_dir}: step {step} written twice with different values")
                pts[int(step)] = val
    return pts, logs


def close(a, b, tol=5e-6):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True,
                    help="the molecule_nat_lang directory holding bert-output/ and gpt2-output/")
    ap.add_argument("--grid-summary", required=True)
    ap.add_argument("--arms-out", required=True)
    ap.add_argument("--curves-out", required=True)
    a = ap.parse_args()

    grid = read_tsv(a.grid_summary)
    if len(grid) != 68:
        raise SystemExit(f"{a.grid_summary}: {len(grid)} arms, expected 68")

    arms = []
    curve_rows = []
    for arch in ("bert", "gpt2"):
        for size in SIZES:
            cands = []
            for g in (r for r in grid if r["arch"] == arch and r["size"] == size):
                name = g["source"].split("/", 1)[0]
                run_dir = os.path.join(a.runs, f"{arch}-output", name)
                if arch == "bert":
                    pts, state = bert_points(run_dir)
                    key = "eval_loss_mask"
                else:
                    pts, logfile = gpt2_points(run_dir)
                    key = "val_loss"
                evals = [(p[key], s) for s, p in pts.items() if key in p]
                if any(not math.isfinite(v) for v, _ in evals):
                    evals = [(v, s) for v, s in evals if math.isfinite(v)]
                best_v, best_s = min(evals)
                cands.append((best_v, best_s, name, run_dir, pts))
            cands.sort()
            best_v, best_s, name, run_dir, pts = cands[0]
            runner_up = cands[1]
            man = json.load(open(os.path.join(run_dir, "run_manifest.json")))

            if arch == "bert":
                ckpt = os.path.join(run_dir, f"checkpoint-{best_s}")
                _, state = bert_points(run_dir)
                if os.path.realpath(state["best_model_checkpoint"]) != os.path.realpath(ckpt):
                    raise SystemExit(f"{name}: best_model_checkpoint "
                                     f"{state['best_model_checkpoint']} is not {ckpt}")
                if not close(state["best_metric"], best_v):
                    raise SystemExit(f"{name}: best_metric {state['best_metric']} != {best_v}")
                # xl is written as shards with an index; the others as one file.
                if not any(os.path.exists(os.path.join(ckpt, f)) for f in
                           ("model.safetensors", "model.safetensors.index.json")):
                    raise SystemExit(f"{ckpt}: no model.safetensors or shard index")
                batch = man["batch"]["effective_global_batch"]
                lr = man["sources"]["learning_rate"]["value"]
                seed = man["seed"]["seed"]
                weights = ckpt
                check = "trainer_state best_model_checkpoint and best_metric"
            else:
                ckpt = os.path.join(run_dir, f"checkpoint-{best_s}")
                targs = json.load(open(os.path.join(ckpt, "training_args.json")))
                if int(targs["iteration"]) != best_s:
                    raise SystemExit(f"{ckpt}: iteration {targs['iteration']} != {best_s}")
                if not close(targs["val_loss"], best_v, tol=5e-4):
                    raise SystemExit(f"{ckpt}: val_loss {targs['val_loss']} != {best_v}")
                batch = man["batch"]["effective_global_batch"]
                lr = man["schedule"]["learning_rate"]
                seed = man["seed"]["seed"]
                weights = os.path.join(ckpt, "training_state.bin")
                check = (f"checkpoint training_args.json iteration and val_loss "
                         f"({targs['val_loss']:.6f}; the CSV rounds to 4 places)")

            arms.append({
                "arm": name, "arch": arch, "size": size, "batch": batch,
                "lr": f"{lr:.3g}", "seed": seed, "checkpoint_step": best_s,
                "judged_metric": "eval_loss_mask" if arch == "bert" else "val_loss",
                "judged_value": f"{best_v:.6f}",
                "runner_up_arm": runner_up[2], "runner_up_value": f"{runner_up[0]:.6f}",
                "manifest_job": man["run"]["job_id"], "manifest_commit": man["run"]["git"]["commit"],
                "checkpoint": os.path.relpath(weights, a.runs),
                "checkpoint_check": check,
            })
            for s in sorted(pts):
                p = pts[s]
                if arch == "bert" and "eval_loss_mask" not in p:
                    continue
                if arch == "gpt2" and "val_loss" not in p:
                    continue
                curve_rows.append({
                    "arm": name, "arch": arch, "size": size, "step": s,
                    "train_loss": f"{p['train_loss']:.6f}" if "train_loss" in p else "",
                    "eval_loss": f"{p['eval_loss' if arch == 'bert' else 'val_loss']:.6f}",
                    "eval_loss_mask": f"{p['eval_loss_mask']:.6f}" if arch == "bert" else "",
                })

    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with open(a.arms_out, "w", encoding="utf-8") as fh:
        fh.write("#! report-tsv 1\n")
        fh.write("# model: several (one row per architecture and size) -- the lowest judged metric of the 2026-11-20 molnl batch-size grid\n")
        fh.write("# source: batch and seed from run_manifest.json; lr from run_manifest.json; judged value and step from trainer_state.json (BERT) / logging_*.csv (GPT-2)\n")
        fh.write(f"# generated: {stamp} by scripts/molnl_chebi20_arms.py\n")
        cols = list(arms[0])
        fh.write("\t".join(cols) + "\n")
        for r in arms:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    with open(a.curves_out, "w", encoding="utf-8") as fh:
        fh.write("#! report-tsv 1\n")
        fh.write("# model: the eight runs of rt1-molnl-chebi20-arms (column arm)\n")
        fh.write("# source: BERT trainer_state.json log_history at the last checkpoint; GPT-2 logging_*.csv\n")
        fh.write("# note: one row per evaluation point, none dropped. BERT train_loss is the HF training loss logged at that step (masked positions only); "
                 "eval_loss is HF's eval_loss and eval_loss_mask the judged metric. GPT-2 eval_loss is val_loss; its CSV carries 4 decimals\n")
        fh.write(f"# generated: {stamp} by scripts/molnl_chebi20_arms.py\n")
        cols = list(curve_rows[0])
        fh.write("\t".join(cols) + "\n")
        for r in curve_rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    for r in arms:
        print("\t".join(str(r[c]) for c in ("arm", "batch", "lr", "seed", "checkpoint_step", "judged_value")))
    print(f"curve rows: {len(curve_rows)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
