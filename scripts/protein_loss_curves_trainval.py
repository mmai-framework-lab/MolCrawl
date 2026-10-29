"""Loss-curve data (train AND validation) for the completed protein best-LR models.

protein-judgment-2026-10-22c §2: one row per evaluation point, a column (`split`)
distinguishing training-data loss from validation-data loss, both included. Sizes
small/medium/large for BERT and GPT-2 now; xl (both) added after completion.

Sources (files only, no GPU):
  * BERT: the newest checkpoint's trainer_state.json log_history. Validation rows come
    from entries with eval_loss_mask (the fixed 10,000-row valid subset); training rows
    from entries carrying the training `loss` (every log_interval). metric names the two.
  * GPT-2: the merged logging_*.csv (iter, train_loss, val_loss). Each logged point emits
    a val row (val_loss, the deterministic fixed eval) and a train row (train_loss).

Columns: run  arch  size  lr  seed  step  split  metric  value  eval_set  job

Run:  python scripts/protein_loss_curves_trainval.py <out_dir>
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

# size -> (run dir name, lr label). BERT large uses the seed-42 re-run (judgment §1).
# xl added after completion (protein-judgment-2026-10-29 §4): BERT xl is the best
# of the six-point LR grid, lr 2.4e-5 (protein_sequence-xl-lr2p4e5).
BERT = {
    "small": ("protein_sequence-small-lr3e4", "3e-4"),
    "medium": ("protein_sequence-medium-lr1e4", "1e-4"),
    "large": ("protein_sequence-large-lr3p5e5-rerun", "3.5e-5"),
    "xl": ("protein_sequence-xl-lr2p4e5", "2.4e-5"),
}
# GPT-2 uses the seed-42 fixed-val re-runs (judgment §1); xl is the seed-42
# fixed-val re-run protein_sequence-ex-large-rerun (protein-judgment-2026-10-29 §4).
GPT2 = {
    "small": ("protein_sequence-small-rerun", "1.2e-3"),
    "medium": ("protein_sequence-medium-rerun", "6e-4"),
    "large": ("protein_sequence-large-rerun", "6e-4"),
    "xl": ("protein_sequence-ex-large-rerun", "6e-4"),
}
COLUMNS = ["run", "arch", "size", "lr", "seed", "step", "split", "metric", "value", "eval_set", "job"]


def newest_ckpt(run_dir: Path):
    steps = [int(os.path.basename(p).split("-")[-1]) for p in glob.glob(str(run_dir / "checkpoint-*"))]
    return max(steps) if steps else None


def manifest_seed(run_dir: Path):
    f = run_dir / "run_manifest.json"
    if not f.exists():
        return ""
    s = json.load(open(f)).get("seed")
    return s.get("seed") if isinstance(s, dict) else s


def manifest_job(run_dir: Path):
    f = run_dir / "run_manifest.json"
    if f.exists():
        d = json.load(open(f))
        if isinstance(d.get("run"), dict):
            return d["run"].get("job_id", "")
    return ""


def bert_segment_jobs(run_dir: Path):
    seg_log = run_dir / "segments.log"
    jobs = []
    if seg_log.exists():
        for line in open(seg_log):
            p = line.split()
            if len(p) >= 2:
                jobs.append(p[1])
    return ";".join(dict.fromkeys(jobs))  # unique, order-preserving


def bert_rows(size, run_name, lr):
    run_dir = BERTO / run_name
    ck = newest_ckpt(run_dir)
    if ck is None:
        return []
    ts = json.load(open(run_dir / f"checkpoint-{ck}" / "trainer_state.json"))
    seed = manifest_seed(run_dir)
    job = bert_segment_jobs(run_dir)
    rows = []
    for e in ts.get("log_history", []):
        if "eval_loss_mask" in e:
            rows.append([run_name, "bert", size, lr, seed, e["step"], "val",
                         "eval_loss_mask", f'{e["eval_loss_mask"]:.6f}', "fixed", job])
        elif "loss" in e and "eval_loss" not in e:
            rows.append([run_name, "bert", size, lr, seed, e["step"], "train",
                         "train_loss", f'{e["loss"]:.6f}', "train_set", job])
    rows.sort(key=lambda r: (r[5], r[6]))
    return rows


def gpt2_rows(size, run_name, lr):
    run_dir = GPT2O / run_name
    seed = manifest_seed(run_dir)
    job = manifest_job(run_dir)
    merged = {}
    for cf in sorted(glob.glob(str(run_dir / "logging_*.csv"))):
        for row in csv.reader(open(cf)):
            if len(row) < 3:
                continue
            try:
                it = int(row[0])
                tr = float(row[1])
                vl = float(row[2])
            except ValueError:
                continue
            merged[it] = (tr, vl)  # later segment overrides
    rows = []
    for it in sorted(merged):
        tr, vl = merged[it]
        rows.append([run_name, "gpt2", size, lr, seed, it, "val", "val_loss", f"{vl:.6f}", "fixed", str(job)])
        rows.append([run_name, "gpt2", size, lr, seed, it, "train", "train_loss", f"{tr:.6f}", "train_set", str(job)])
    return rows


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/protein_loss_curves_trainval.py <out_dir>")
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "protein-loss-curves-trainval.tsv"

    all_rows = []
    for size, (run_name, lr) in BERT.items():
        all_rows += bert_rows(size, run_name, lr)
    for size, (run_name, lr) in GPT2.items():
        all_rows += gpt2_rows(size, run_name, lr)

    with open(tsv, "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in all_rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    # summary
    from collections import Counter
    c = Counter((r[1], r[2], r[6]) for r in all_rows)
    print(f"wrote {tsv}  ({len(all_rows)} rows)")
    for (arch, size, split) in sorted(c):
        print(f"  {arch:5s} {size:6s} {split:5s}: {c[(arch, size, split)]} points")


if __name__ == "__main__":
    main()
