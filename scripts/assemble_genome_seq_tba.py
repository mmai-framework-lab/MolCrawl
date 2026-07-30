"""Assemble genome_sequence TBA metrics CSV (hybrid 20260316 / 20260520_uncapped).

Implements `genome_seq_tba_metrics_instructions.md` with the hybrid source
plan agreed with matsubara:
  - GPT-2 small  : 20260316           (20260520 LR-shock degraded)
  - GPT-2 medium : 20260520_uncapped  (improved over 20260316)
  - GPT-2 large  : 20260520_uncapped  (improved over 20260316)
  - BERT small   : 20260316           (~equiv, integrity)
  - DNABERT2 m/l : 20260316           (large absent in 20260520)

Outputs:
  tmp/metrics_genome_seq_tba.csv     -- the 6-row TBA table with `source_dir`
                                       column for traceability.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO = Path("/lustre/home/matsubara/riken-dataset-fundational-model")
SUMMARY_CSV = REPO / "tmp" / "metrics_summary.csv"
MLM_ROOT = REPO / "tmp" / "mlm_accuracy"
OUT_CSV = REPO / "tmp" / "metrics_genome_seq_tba.csv"

# Hybrid plan (model, arch, size, source_dir).
PLAN = [
    ("GPT-2",    "small",  "learning_source_20260316"),
    ("GPT-2",    "medium", "learning_source_20260520_uncapped"),
    ("GPT-2",    "large",  "learning_source_20260520_uncapped"),
    ("BERT",     "small",  "learning_source_20260316"),
    ("DNABERT2", "medium", "learning_source_20260316"),
    ("DNABERT2", "large",  "learning_source_20260316"),
]

# Training hyper-parameter table (from configs/genome_sequence/*.py).
# n_GPUs is the workflows/03* default (the one that was actually used for
# the launched runs; verified via the wrapper's `NUM_GPUS=${NUM_GPUS:-N}`).
HPARAM = {
    ("GPT-2",    "small"):  dict(lr=6e-6, batch=12, grad_acc=40, n_gpus=8, max_len=1024, max_steps=50000, warmup=200),
    ("GPT-2",    "medium"): dict(lr=6e-6, batch=12, grad_acc=40, n_gpus=8, max_len=1024, max_steps=50000, warmup=200),
    ("GPT-2",    "large"):  dict(lr=6e-6, batch=12, grad_acc=40, n_gpus=8, max_len=1024, max_steps=50000, warmup=200),
    ("BERT",     "small"):  dict(lr=6e-6, batch=8,  grad_acc=80, n_gpus=1, max_len=1024, max_steps=60000, warmup=0),
    ("DNABERT2", "medium"): dict(lr=3e-5, batch=16, grad_acc=4,  n_gpus=2, max_len=1024, max_steps=200000, warmup=0),
    ("DNABERT2", "large"):  dict(lr=3e-5, batch=16, grad_acc=4,  n_gpus=4, max_len=1024, max_steps=200000, warmup=0),
}

# Map "20260316" / "20260520_uncapped" → mlm_accuracy.json label path
MLM_LABEL = {
    ("BERT",     "small",  "20260316"): "genome_seq_tba__bert-small-20260316",
    ("DNABERT2", "medium", "20260316"): "genome_seq_tba__dnabert2-medium-20260316",
    ("DNABERT2", "large",  "20260316"): "genome_seq_tba__dnabert2-large-20260316",
}


def load_summary_row(src: str, arch: str, size: str):
    """Return dict row from metrics_summary.csv matching the spec, else None."""
    with SUMMARY_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["source_dir"] == src
                    and row["modality"] == "genome_sequence"
                    and row["arch"] == arch
                    and row["size"] == size
                    and row.get("variant", "base") == "base"):
                return row
    return None


def load_mlm_acc(arch: str, size: str, src_tag: str):
    label = MLM_LABEL.get((arch, size, src_tag))
    if label is None:
        return None
    p = MLM_ROOT / label / "mlm_accuracy.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("mlm_accuracy")


def tokens_seen_M(steps: int, hp: dict, arch: str) -> float:
    # GPT-2 is nanoGPT (this repo): `gradient_accumulation_steps = 5 * n_GPUs`
    # is the GLOBAL grad accum, so n_gpus is already baked in — DO NOT multiply
    # again. BERT/DNABERT2 use HF Trainer where grad_accum is per-device, so
    # the n_gpus factor IS required.
    if arch == "GPT-2":
        return steps * hp["batch"] * hp["grad_acc"] * hp["max_len"] / 1e6
    return steps * hp["batch"] * hp["grad_acc"] * hp["n_gpus"] * hp["max_len"] / 1e6


def main() -> int:
    rows = []
    for arch, size, src in PLAN:
        src_tag = src.replace("learning_source_", "").replace("_uncapped", "")
        # Note: src_tag for "learning_source_20260520_uncapped" → "20260520"
        sr = load_summary_row(src, arch, size)
        if sr is None:
            print(f"[WARN] no summary row for ({src}, {arch}, {size})")
            continue
        val_loss = float(sr["val_loss"]) if sr["val_loss"] else None
        steps = int(sr["steps"]) if sr["steps"] else None
        hp = HPARAM[(arch, size)]
        perp = math.exp(val_loss) if val_loss is not None else None
        bpc = val_loss / math.log(2) if val_loss is not None else None
        tok_M = tokens_seen_M(steps, hp, arch) if steps is not None else None
        mlm = load_mlm_acc(arch, size, src_tag)
        eff_batch = hp["batch"] * hp["grad_acc"] * hp["n_gpus"]
        rows.append({
            "model": "genome_sequence",
            "arch": arch,
            "size": size,
            "source_dir": src,
            "val_loss": f"{val_loss:.4f}" if val_loss is not None else "",
            "perplexity": f"{perp:.2f}" if perp is not None else "",
            "BPC": f"{bpc:.4f}" if bpc is not None else "",
            "steps": steps if steps is not None else "",
            "tokens_seen_M": f"{tok_M:.0f}" if tok_M is not None else "",
            "mlm_acc": f"{mlm:.4f}" if mlm is not None else "",
            "lr": hp["lr"],
            "batch_size": hp["batch"],
            "grad_acc": hp["grad_acc"],
            "n_gpus": hp["n_gpus"],
            "eff_batch_seq": eff_batch,
            "max_len": hp["max_len"],
            "max_steps": hp["max_steps"],
            "warmup": hp["warmup"],
        })
        print(f"  [{arch:<8} {size:<7}] src={src_tag:<10}  "
              f"val={val_loss:.4f}  steps={steps:<6}  "
              f"perp={perp:>9.2f}  BPC={bpc:.4f}  "
              f"tok={tok_M:>8.0f}M  mlm={mlm if mlm is None else f'{mlm:.4f}'}")

    if not rows:
        print("[ERROR] no rows produced", file=sys.stderr)
        return 1

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] wrote {OUT_CSV}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
