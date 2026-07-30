"""Assemble the boss-spec paper-ready metrics tables.

Reads:
  - tmp/metrics_summary.csv (from collect_metrics_discovery.py)
  - tmp/mlm_accuracy/<label>/mlm_accuracy.json (from eval_bert_mlm_accuracy.py)
  - molcrawl/tasks/pretrain/configs/{modality}/bert_*.py (for BERT
    tokens_seen formula)

Writes:
  - tmp/metrics_main.csv         (paper main table: 4 modality × {GPT-2, BERT}
                                  × {small, medium, large} + genome mammal_centered)
  - tmp/metrics_genome_subsets.csv (supplement: 20 Evo2 subsets × {GPT-2, BERT})

Column schema (per boss spec):
  modality, arch, size, val_loss, perplexity_or_pseudo_ppl,
  BPC, steps, tokens_seen_M, mlm_accuracy, src
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Optional

REPO = Path("/lustre/home/matsubara/riken-dataset-fundational-model")
SUMMARY_CSV = REPO / "tmp" / "metrics_summary.csv"
MLM_DIR     = REPO / "tmp" / "mlm_accuracy"
OUT_MAIN    = REPO / "tmp" / "metrics_main.csv"
OUT_SUBSET  = REPO / "tmp" / "metrics_genome_subsets.csv"

PRIORITY = [
    "learning_source_20260520_uncapped",
    "learning_source_20260316",
    "learning_source_20260407_compounds",
    "learning_source_20260302",
]
SUBSET_SRC = "learning_source_20260529_evo2species"
TARGET_MODALITIES = ["protein_sequence", "rna", "compounds", "molecule_nat_lang"]
TARGET_SIZES = ["small", "medium", "large"]
SUBSETS = ([
    "mammal_centered",
] + [f"eukaryote_matched_random_seed{i}" for i in range(1, 11)]
   + [f"global_random_seed{i}" for i in range(1, 11)])


def load_mlm_accuracies() -> dict[str, dict]:
    """Map "<modality>/<size>" → mlm_accuracy.json contents."""
    out = {}
    if not MLM_DIR.exists():
        return out
    for p in MLM_DIR.glob("*/mlm_accuracy.json"):
        try:
            rec = json.loads(p.read_text())
            out[rec["label"]] = rec
        except Exception:
            pass
    return out


def bert_tokens_seen_estimate(
    modality: str, size: str, steps: int,
    num_gpus_default: int = 4,
) -> Optional[int]:
    """Formula: steps × per_device_batch × grad_accum × num_gpus × max_seq_length.

    Pulls the constants from molcrawl/tasks/pretrain/configs/<modality>/bert_<size>.py.
    """
    cfg_path = REPO / "molcrawl" / "tasks" / "pretrain" / "configs" / modality / f"bert_{size}.py"
    if not cfg_path.exists():
        return None
    src = cfg_path.read_text()
    def grab(key, default=None):
        m = re.search(rf"^\s*{key}\s*=\s*(\d+)", src, re.MULTILINE)
        return int(m.group(1)) if m else default
    bs = grab("batch_size") or grab("per_device_train_batch_size")
    ga = grab("gradient_accumulation_steps", 1)
    ml = grab("max_length") or grab("max_seq_length", 512)
    if bs is None or ml is None:
        return None
    return int(steps * bs * ga * num_gpus_default * ml)


def fmt(v, d=4):
    if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def pick_canonical(rows: list[dict], modality: str, arch: str, size: str,
                   variant: str = "base") -> Optional[dict]:
    """Pick the highest-priority source row matching (modality, arch, size, variant)."""
    best = None
    best_idx = 99
    for r in rows:
        if r["modality"] != modality:
            continue
        if r["arch"] != arch:
            continue
        if r["size"] != size:
            continue
        if r["variant"] != variant:
            continue
        if r["source_dir"] not in PRIORITY:
            continue
        i = PRIORITY.index(r["source_dir"])
        if i < best_idx:
            best, best_idx = r, i
    return best


def pick_subset(rows: list[dict], subset: str, arch: str) -> Optional[dict]:
    """For genome subset campaign, variant == subset name (with full prefix)."""
    for r in rows:
        if r["source_dir"] != SUBSET_SRC:
            continue
        if r["arch"] != arch:
            continue
        if r["variant"] == subset:
            return r
    return None


def assemble_row(r: dict, mlm_lookup: dict, label: str, override_modality: str = None) -> dict:
    """Build one paper-row from a metrics_summary.csv row + mlm accuracy lookup."""
    modality = override_modality if override_modality else r["modality"]
    arch = r["arch"]
    size = r["size"]
    val_loss = r["val_loss"] or ""
    perp = r["perplexity"] or r["pseudo_perp"] or ""
    bpc = r["bpc"] or ""
    steps = r["steps"] or ""
    tokens = r["tokens_seen_M"] or ""

    # BERT tokens — estimate via formula if empty
    if arch == "BERT" and not tokens and steps:
        try:
            est = bert_tokens_seen_estimate(modality, size, int(steps))
            if est:
                tokens = f"{est/1e6:.0f} (estimated)"
        except Exception:
            pass

    # MLM accuracy — only for BERT
    mlm_acc = ""
    if arch == "BERT" and label in mlm_lookup:
        v = mlm_lookup[label].get("mlm_accuracy")
        if v is not None:
            mlm_acc = f"{v:.4f}"
        elif mlm_lookup[label].get("error"):
            mlm_acc = "(error)"

    return {
        "modality": modality,
        "arch": arch,
        "size": size,
        "val_loss": fmt(float(val_loss)) if val_loss else "",
        "perplexity_or_pseudo_ppl": perp,
        "BPC": bpc,
        "steps": steps,
        "tokens_seen_M": tokens,
        "mlm_accuracy": mlm_acc,
        "src": r["source_dir"].replace("learning_source_", "ls_"),
    }


def main() -> int:
    rows = list(csv.DictReader(SUMMARY_CSV.open()))
    mlm = load_mlm_accuracies()
    print(f"loaded {len(rows)} summary rows, {len(mlm)} mlm_accuracy results")

    # === Main table ===
    main_rows: list[dict] = []
    for modality in TARGET_MODALITIES:
        for arch in ("GPT-2", "BERT"):
            for size in TARGET_SIZES:
                r = pick_canonical(rows, modality, arch, size, variant="base")
                if r is None:
                    print(f"  MISS: {modality}/{arch}/{size}")
                    continue
                label = f"{modality}/{size}"
                main_rows.append(assemble_row(r, mlm, label))
    # genome_sequence — mammal_centered のみ
    for arch in ("GPT-2", "BERT"):
        r = pick_subset(rows, "mammal_centered", arch)
        if r is None:
            print(f"  MISS: genome_sequence/mammal_centered/{arch}")
            continue
        label = "genome/mammal_centered"
        main_rows.append(assemble_row(r, mlm, label, override_modality="genome_sequence (mammal_centered)"))

    fields = ["modality", "arch", "size", "val_loss",
              "perplexity_or_pseudo_ppl", "BPC", "steps",
              "tokens_seen_M", "mlm_accuracy", "src"]
    with OUT_MAIN.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(main_rows)
    print(f"wrote {OUT_MAIN}  ({len(main_rows)} rows)")

    # === Supplement: 20 Evo2 subset (mammal_centered 除く) ===
    sub_rows: list[dict] = []
    for subset in SUBSETS:
        if subset == "mammal_centered":
            continue
        for arch in ("GPT-2", "BERT"):
            r = pick_subset(rows, subset, arch)
            if r is None:
                print(f"  MISS: genome/{subset}/{arch}")
                continue
            label = f"genome/{subset}"
            row = assemble_row(r, mlm, label, override_modality=subset)
            # 補足表は modality 列を "subset" 列に rename
            row = {
                "subset": subset, "arch": row["arch"], "size": row["size"],
                "val_loss": row["val_loss"],
                "perplexity_or_pseudo_ppl": row["perplexity_or_pseudo_ppl"],
                "BPC": row["BPC"], "steps": row["steps"],
                "tokens_seen_M": row["tokens_seen_M"],
                "mlm_accuracy": row["mlm_accuracy"], "src": row["src"],
            }
            sub_rows.append(row)
    sub_fields = ["subset", "arch", "size", "val_loss",
                  "perplexity_or_pseudo_ppl", "BPC", "steps",
                  "tokens_seen_M", "mlm_accuracy", "src"]
    with OUT_SUBSET.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sub_fields)
        w.writeheader()
        w.writerows(sub_rows)
    print(f"wrote {OUT_SUBSET}  ({len(sub_rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
