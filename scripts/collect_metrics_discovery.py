"""Discovery-based variant of the boss-supplied collect_metrics.py.

The original script in collect_training_metrics_instructions.md assumed a
single LEARNING_SOURCE_DIR with a fixed (modality, arch, size, variant)
directory layout. The actual layout in /lustre/home/matsubara/
deviates in 4 ways:

  1. Multiple learning_source_* dirs (20260316 / 20260407_compounds /
     20260520_uncapped / 20260529_evo2species) hold different
     modalities, with overlap.
  2. The "-{variant}" suffix is sometimes realised as a sibling parent
     directory (e.g. ``genome_sequence_clinvar/``) rather than a suffix
     on the model-output dir.
  3. HuggingFace Trainer writes trainer_state.json *per checkpoint*
     (e.g. ``checkpoint-60000/trainer_state.json``), not flat under the
     model output dir.
  4. The boss used ``mol_nat_lang`` but the actual dir name is
     ``molecule_nat_lang``.

This script discovers all ``ckpt.pt`` and the *latest* per-output-dir
``checkpoint-*/trainer_state.json``, extracts the same metric set
(val_loss, perplexity, pseudo-perplexity, BPC, MLM accuracy, steps,
tokens_seen) and writes the same CSV columns expected by the boss.

Usage:
  python tmp/scripts/collect_metrics_discovery.py \
      [--source-dirs A,B,C] \
      [--out metrics_summary.csv]

Defaults to scanning every ``/lustre/home/matsubara/learning_source_*``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from pathlib import Path


DEFAULT_SOURCE_DIRS = sorted(glob.glob("/lustre/home/matsubara/learning_source_*"))


_SIZE_RE = re.compile(
    r"-(?P<size>x?l?small|small|medium|large|ex-large|xl)\b", re.IGNORECASE
)


def parse_arch(dir_name: str) -> str:
    """E.g. 'bert-output' → 'BERT', 'chemberta2-output' → 'ChemBERTa2'."""
    arch = dir_name.replace("-output", "")
    table = {
        "bert": "BERT", "gpt2": "GPT-2", "dnabert2": "DNABERT2",
        "esm2": "ESM-2", "chemberta2": "ChemBERTa2",
        "llama": "LLaMA", "roberta": "RoBERTa", "rnaformer": "RNAformer",
    }
    return table.get(arch.lower(), arch)


def parse_size_and_suffix(
    model_subdir: str,
    modality_prefix: str,
    arch_prefix: str = "",
) -> tuple[str, str]:
    """E.g. 'genome_sequence-small-clinvar' → ('small', 'clinvar').

    Strips whichever known prefix matches (modality or arch). HuggingFace
    output dirs sometimes name the leaf by arch instead of modality, e.g.
    ``compounds/chemberta2-output/chemberta2-medium/`` where the leaf is
    ``chemberta2-medium`` (arch-named) rather than ``compounds-medium``
    (modality-named).
    """
    stripped = model_subdir
    for pfx in (modality_prefix, arch_prefix):
        if pfx and stripped.startswith(pfx + "-"):
            stripped = stripped[len(pfx) + 1:]
            break
    parts = stripped.split("-")
    size = parts[0] if parts else ""
    variant = "-".join(parts[1:]) if len(parts) > 1 else "base"
    # Accept multi-word sizes like 'ex-large'
    if len(parts) >= 2 and parts[0] == "ex" and parts[1] == "large":
        size = "ex-large"
        variant = "-".join(parts[2:]) if len(parts) > 2 else "base"
    return size, variant


def is_backup(variant: str, size: str, model_dir_name: str) -> bool:
    """Filter out backup / broken / immature retraining artifacts."""
    BACKUP_MARKERS = (
        "_backup", "_pre_retokenize", "_overfit", "_patience5",
        "_old_immature_pretrain", "broken_vocab",
        "yigarashi",  # one-shot initial test dir, not a tracked model
    )
    blob = " ".join([variant, size, model_dir_name])
    return any(m in blob for m in BACKUP_MARKERS)


def latest_trainer_state(model_dir: Path) -> Path | None:
    """Find the highest-numbered checkpoint-*/trainer_state.json under model_dir."""
    candidates: list[tuple[int, Path]] = []
    for p in model_dir.glob("checkpoint-*/trainer_state.json"):
        m = re.search(r"checkpoint-(\d+)", p.parent.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def metrics_from_ckpt(ckpt_path: Path) -> dict:
    """Pull val_loss/iter/cfg from a nanoGPT ckpt.pt.

    Uses mmap=True so the (potentially multi-GB) model + optimizer
    tensors are memory-mapped instead of fully read into RAM — we only
    touch the scalar metadata fields (best_val_loss, iter_num, config).
    On a login node with limited RAM, a non-mmap full load of a single
    GPT-2 large/xl ckpt.pt was enough to get the process OOM-killed
    mid-scan.
    """
    import torch
    try:
        # mmap=True requires weights_only=True on modern PyTorch.
        ckpt = torch.load(
            str(ckpt_path), map_location="cpu",
            weights_only=True, mmap=True,
        )
    except TypeError:
        # Older PyTorch without mmap kwarg — fall back to weights_only.
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except Exception as e:
            return {"error": f"torch.load failed: {e}"}
    except Exception as e:
        return {"error": f"torch.load failed: {e}"}
    try:
        val_loss = float(ckpt.get("best_val_loss", float("nan")))
        iter_num = int(ckpt.get("iter_num", -1))
        cfg = ckpt.get("config", {})
        bs    = cfg.get("batch_size") if isinstance(cfg, dict) else None
        blk   = cfg.get("block_size") if isinstance(cfg, dict) else None
        grad  = cfg.get("gradient_accumulation_steps", 1) if isinstance(cfg, dict) else 1
    finally:
        # Drop references to mmapped tensors as soon as possible.
        del ckpt
    tokens = None
    if bs and blk and iter_num > 0:
        tokens = iter_num * bs * blk * grad
    return {
        "val_loss": val_loss, "iter_num": iter_num,
        "batch_size": bs, "block_size": blk, "grad_accum": grad,
        "tokens_seen": tokens,
    }


def metrics_from_trainer_state(state_path: Path) -> dict:
    """Pull eval_loss / step / eval_accuracy from a HuggingFace trainer_state.json."""
    try:
        with state_path.open() as f:
            state = json.load(f)
    except Exception as e:
        return {"error": f"json load failed: {e}"}
    history = state.get("log_history", [])
    eval_entries = [e for e in history if "eval_loss" in e]
    if not eval_entries:
        return {"error": "no eval_loss in log_history"}
    last = eval_entries[-1]
    eval_loss = float(last["eval_loss"])
    step = int(last.get("step", state.get("global_step", -1)))
    # MLM accuracy may live under several keys; only some training scripts
    # implement compute_metrics for MLM.
    acc = (
        last.get("eval_accuracy")
        or last.get("eval_masked_lm_accuracy")
        or last.get("eval_acc")
    )
    return {
        "eval_loss": eval_loss, "step": step,
        "mlm_accuracy": float(acc) if acc is not None else None,
    }


def scan_modality(source_dir: Path, modality_dir: Path) -> list[dict]:
    """Walk a single ``<source>/<modality>/<arch>-output/<model>/`` tree."""
    rows: list[dict] = []
    src_tag = source_dir.name
    # The modality_dir name might be e.g. 'genome_sequence',
    # 'genome_sequence_clinvar', 'protein_sequence_proteingym' etc.
    full_mod = modality_dir.name
    # Split off the trailing variant ('genome_sequence_clinvar' →
    # base modality 'genome_sequence', dataset variant 'clinvar').
    BASE_MODS = (
        "genome_sequence", "protein_sequence", "rna",
        "compounds", "molecule_nat_lang",
    )
    base_modality = full_mod
    parent_variant = "base"
    for bm in BASE_MODS:
        if full_mod == bm:
            base_modality, parent_variant = bm, "base"
            break
        if full_mod.startswith(bm + "_"):
            base_modality = bm
            parent_variant = full_mod[len(bm) + 1:]
            break

    for arch_dir in sorted(modality_dir.glob("*-output")):
        if not arch_dir.is_dir():
            continue
        arch_str = parse_arch(arch_dir.name)
        arch_prefix = arch_dir.name.replace("-output", "")  # e.g. 'chemberta2'
        for model_dir in sorted(arch_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            size, suffix_variant = parse_size_and_suffix(
                model_dir.name,
                base_modality if base_modality == full_mod else full_mod,
                arch_prefix,
            )
            # Combine the parent_variant (dataset fork) with any per-output
            # suffix (-clinvar / -proteingym etc.) for the final variant tag.
            if parent_variant != "base" and suffix_variant != "base":
                variant = f"{parent_variant}__{suffix_variant}"
            elif parent_variant != "base":
                variant = parent_variant
            else:
                variant = suffix_variant or "base"

            # Skip backup / broken / immature retraining artefacts.
            if is_backup(variant, size, model_dir.name):
                continue

            row = {
                "source_dir": src_tag,
                "modality": base_modality,
                "arch": arch_str,
                "size": size,
                "variant": variant,
                "model_dir": str(model_dir),
                "val_loss": None, "perplexity": None, "pseudo_perp": None,
                "bpc": None, "mlm_accuracy": None,
                "steps": None, "tokens_seen_M": None, "source": "NOT FOUND",
            }
            ckpt = model_dir / "ckpt.pt"
            ts   = latest_trainer_state(model_dir)
            if ckpt.exists():
                m = metrics_from_ckpt(ckpt)
                if "error" in m:
                    row["source"] = f"ckpt.pt(error): {m['error']}"
                else:
                    vl = m["val_loss"]
                    row["val_loss"]      = vl
                    row["perplexity"]    = math.exp(min(vl, 50)) if not math.isnan(vl) else None
                    row["bpc"]           = vl / math.log(2) if not math.isnan(vl) else None
                    row["steps"]         = m["iter_num"]
                    if m["tokens_seen"]:
                        row["tokens_seen_M"] = m["tokens_seen"] / 1e6
                    row["source"] = "ckpt.pt"
            elif ts is not None:
                m = metrics_from_trainer_state(ts)
                if "error" in m:
                    row["source"] = f"trainer_state(error): {m['error']}"
                else:
                    el = m["eval_loss"]
                    row["val_loss"]      = el
                    row["pseudo_perp"]   = math.exp(min(el, 50))
                    row["bpc"]           = el / math.log(2)
                    row["mlm_accuracy"]  = m["mlm_accuracy"]
                    row["steps"]         = m["step"]
                    row["source"] = f"trainer_state.json ({ts.parent.name})"
            else:
                # No metrics file at all. Still emit the row so the boss
                # can see what model dirs exist but have no training
                # artefact yet.
                row["source"] = "no ckpt.pt or checkpoint-*/trainer_state.json"

            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source-dirs", default=",".join(DEFAULT_SOURCE_DIRS),
        help="Comma-separated list of source dirs to scan",
    )
    ap.add_argument(
        "--out",
        default="/lustre/home/matsubara/riken-dataset-fundational-model/tmp/metrics_summary.csv",
    )
    args = ap.parse_args()

    source_dirs = [Path(p) for p in args.source_dirs.split(",") if p]
    all_rows: list[dict] = []
    for src in source_dirs:
        if not src.exists():
            continue
        for mod_dir in sorted(src.iterdir()):
            if not mod_dir.is_dir():
                continue
            # Heuristic: only descend into dirs that look like modality
            # roots (contain at least one *-output sibling).
            if not any(d.name.endswith("-output") for d in mod_dir.iterdir() if d.is_dir()):
                continue
            all_rows.extend(scan_modality(src, mod_dir))

    # Fill in formatted strings for output
    def fmt(v, d=4):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if isinstance(v, float):
            return f"{v:.{d}f}"
        return str(v)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_dir", "modality", "arch", "size", "variant",
        "val_loss", "perplexity", "pseudo_perp", "bpc", "mlm_accuracy",
        "steps", "tokens_seen_M", "source", "model_dir",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({
                "source_dir":    r["source_dir"],
                "modality":      r["modality"],
                "arch":          r["arch"],
                "size":          r["size"],
                "variant":       r["variant"],
                "val_loss":      fmt(r["val_loss"]),
                "perplexity":    fmt(r["perplexity"], 2),
                "pseudo_perp":   fmt(r["pseudo_perp"], 2),
                "bpc":           fmt(r["bpc"]),
                "mlm_accuracy":  fmt(r["mlm_accuracy"]),
                "steps":         fmt(r["steps"], 0),
                "tokens_seen_M": fmt(r["tokens_seen_M"], 0),
                "source":        r["source"],
                "model_dir":     r["model_dir"],
            })

    print(f"\nwrote {out_path}  ({len(all_rows)} rows)")

    # Console preview — group by modality+arch
    print()
    print(f"{'src':<12} {'mod':<18} {'arch':<10} {'size':<10} {'variant':<28} "
          f"{'val_loss':>9} {'ppl/pseudo':>12} {'bpc':>7} {'acc':>7} {'steps':>7} {'source':<20}")
    print("-" * 160)
    for r in all_rows:
        ppl = (r["perplexity"] if r["perplexity"] is not None else r["pseudo_perp"])
        ppl_s = f"{ppl:.2f}" if isinstance(ppl, float) and not math.isnan(ppl) else "-"
        vl_s  = f"{r['val_loss']:.4f}" if isinstance(r['val_loss'], float) and not math.isnan(r['val_loss']) else "-"
        bpc_s = f"{r['bpc']:.4f}" if isinstance(r['bpc'], float) and not math.isnan(r['bpc']) else "-"
        acc_s = f"{r['mlm_accuracy']:.4f}" if isinstance(r['mlm_accuracy'], float) else "-"
        steps_s = str(r['steps']) if r['steps'] is not None else "-"
        src_short = r["source_dir"].replace("learning_source_", "ls_")[:12]
        src_label = r["source"][:20]
        print(f"{src_short:<12} {r['modality']:<18} {r['arch']:<10} {r['size']:<10} "
              f"{r['variant']:<28} {vl_s:>9} {ppl_s:>12} {bpc_s:>7} {acc_s:>7} {steps_s:>7} {src_label:<20}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
