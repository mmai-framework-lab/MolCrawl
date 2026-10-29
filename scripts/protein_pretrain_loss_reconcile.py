"""Reconcile the five protein pretrain-loss rows where the reported ladder (§1 of
protein-progress-and-decision-2026-10-27) disagrees with the distributed
proteingym-per-model.tsv pretrain_loss column (protein-judgment-2026-10-29 §5).

For each of the five models (GPT-2 small/medium/large/xl and BERT large) it emits
one row per value -- the reported §1 value and the distributed-TSV value -- read
straight from the runs on disk (no hard-coded numbers), with the run, the checkpoint
step the value sits at, the validation set it was measured on, the value itself, and
whether it is the adopted (confirmed) value.

Both GPT-2 values are the minimum val_loss over the run's merged logging_*.csv; both
BERT values are the minimum eval_loss_mask over the newest checkpoint's trainer_state
log_history -- the same reductions the two sources used. The only differences are the
run (seed), the validation set, and the step the minimum lands at.

Run:  python scripts/protein_pretrain_loss_reconcile.py <out_dir>
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BERTO = REPO / "learning_source_protein_runs/protein_sequence/bert-output"
GPT2O = REPO / "learning_source_protein_runs/protein_sequence/gpt2-output"
RUNS_U50 = REPO / "learning_source_20260730_protein_uniref50/protein_sequence/runs"

COLUMNS = ["arch", "size", "source", "run", "seed", "checkpoint_step",
           "validation_set", "value", "adopted", "note"]

# arch, size, source label, run dir, validation-set description, adopted, note
ROWS = [
    ("gpt2", "small", "report_s1", GPT2O / "protein_sequence-small-rerun",
     "valid 固定 (seed42・決定的 held-out、学習中の val_loss)", "yes", "確定 (seed42・検証集合固定)"),
    ("gpt2", "small", "distributed_tsv", GPT2O / "protein_sequence-small-lr1p2e3",
     "valid 訓練時 resampled (production 走の val_loss 最小)", "no", "seed1001 の旧 production 走・採らない"),
    ("gpt2", "medium", "report_s1", GPT2O / "protein_sequence-medium-rerun",
     "valid 固定 (seed42・決定的 held-out、学習中の val_loss)", "yes", "確定 (seed42・検証集合固定)"),
    ("gpt2", "medium", "distributed_tsv", RUNS_U50 / "retrain2-gpt2-medium-lr6e-4-s1001",
     "valid 訓練時 resampled (seed1001 retrain2 の val_loss 最小)", "no", "seed1001・resampled val 最小 (最良 step 30150) で楽観側・採らない"),
    ("gpt2", "large", "report_s1", GPT2O / "protein_sequence-large-rerun",
     "valid 固定 (seed42・決定的 held-out、学習中の val_loss)", "yes", "確定 (seed42・検証集合固定)"),
    ("gpt2", "large", "distributed_tsv", RUNS_U50 / "retrain2-gpt2-large-lr6e-4-s1001",
     "valid 訓練時 resampled (seed1001 retrain2 の val_loss 最小)", "no", "seed1001・resampled val 最小 (最良 step 30150) で楽観側・採らない"),
    ("gpt2", "xl", "report_s1", GPT2O / "protein_sequence-ex-large-rerun",
     "valid 固定 (seed42・決定的 held-out、学習中の val_loss)", "yes", "確定 (seed42・検証集合固定)"),
    ("gpt2", "xl", "distributed_tsv", RUNS_U50 / "retrain2-gpt2-xl-lr6e-4-s1001",
     "valid 訓練時 resampled (seed1001 retrain2 の val_loss 最小)", "no", "seed1001・resampled val 最小・採らない"),
    ("bert", "large", "report_s1", BERTO / "protein_sequence-large-lr3p5e5-rerun",
     "valid 固定 (eval_loss_mask・固定マスク)", "yes", "確定 (seed42・検証集合固定の再走)"),
    ("bert", "large", "distributed_tsv", BERTO / "protein_sequence-large-lr3p5e5",
     "valid (production 走・固定マスク化前の eval_loss_mask 最小)", "no", "固定マスク化前の production 評価・採らない"),
]


def manifest_seed(run_dir: Path):
    f = run_dir / "run_manifest.json"
    if not f.exists():
        return ""
    s = json.load(open(f)).get("seed")
    if isinstance(s, dict):
        return s.get("seed", "")
    return s if s is not None else ""


def gpt2_min(run_dir: Path):
    best = (1e9, None)
    for cf in sorted(glob.glob(str(run_dir / "logging_*.csv"))):
        for row in csv.reader(open(cf)):
            if len(row) < 3:
                continue
            try:
                it = int(row[0])
                vl = float(row[2])
            except ValueError:
                continue
            if vl < best[0]:
                best = (vl, it)
    return (best[0] if best[1] is not None else None), best[1]


def bert_min(run_dir: Path):
    cks = sorted((int(os.path.basename(p).split("-")[-1]) for p in glob.glob(str(run_dir / "checkpoint-*"))), reverse=True)
    for ck in cks:
        f = run_dir / f"checkpoint-{ck}" / "trainer_state.json"
        if f.exists():
            ts = json.load(open(f))
            pts = [(x.get("step"), x["eval_loss_mask"]) for x in ts.get("log_history", []) if "eval_loss_mask" in x]
            if pts:
                mn = min(pts, key=lambda t: t[1])
                return mn[1], mn[0]
    return None, None


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/protein_pretrain_loss_reconcile.py <out_dir>")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    tsv = out / "protein-pretrain-loss-reconcile.tsv"

    with open(tsv, "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for arch, size, source, run_dir, valset, adopted, note in ROWS:
            seed = manifest_seed(run_dir)
            if arch == "gpt2":
                val, step = gpt2_min(run_dir)
            else:
                val, step = bert_min(run_dir)
            val_s = f"{val:.4f}" if val is not None else ""
            fh.write("\t".join(str(x) for x in [
                arch, size, source, run_dir.name, seed,
                step if step is not None else "", valset, val_s, adopted, note]) + "\n")
    print(f"wrote {tsv}")


if __name__ == "__main__":
    main()
