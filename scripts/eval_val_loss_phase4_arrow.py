"""Evaluate val_loss / val_perplexity for every (subset, model) ckpt against
the post-Phase-4-fix Arrow valid split.

Why this script exists
----------------------
Before the Phase 4 split shuffle fix (PR #65), each subset's valid / test
came from the alphabetically-last parquet, so a single multi-Gbp eukaryote
genome (e.g. Balaenoptera acutorostrata) dominated the loss across
multiple subsets and the val_loss recorded during training was not a fair
subset-comparison metric.

After the fix + the Phase 4 rerun (PR #72), each subset now has a
shuffled, whale-free valid split, but the ckpts on disk were trained
against the OLD splits and never had their val_loss re-measured against
the NEW splits. This script does exactly that — load each existing ckpt
and forward-pass it over the NEW Arrow valid set, with no gradient and
no weight updates, producing a clean per-(subset, model) val_loss number
that *is* fair for subset comparison.

The script writes:
  <LEARNING_SOURCE_DIR>/genome_sequence/analysis/val_loss_phase4_fixed/
    val_loss_results.csv        # subset, model, n_batches, val_loss, val_ppl
    val_loss_report.txt          # human-readable table
    per_run/{model}__{subset}/   # tiny run.log per ckpt

Usage:
    export LEARNING_SOURCE_DIR=/path/to/learning_source
    python scripts/eval_val_loss_phase4_arrow.py [--dry-run] [--subsets ...]
        [--models bert,gpt2] [--device cuda|cpu] [--eval-iters 200]

Per-model batch sources:
- GPT-2 (nanoGPT format checkpoint-50000/training_state.bin):
    The Arrow dataset's "valid" split is a Dataset({"input_ids": [int]}).
    We pack consecutive token blocks of size block_size=context_length and
    compute next-token CE loss using nanoGPT's GPT.forward(targets=...)
    interface, then average over up to --eval-iters random blocks.
- BERT MLM (HuggingFace format checkpoint-60000):
    We sample blocks from the valid split, apply the same dynamic mask
    used at training time (15 %, mlm_probability=0.15 by default), and
    compute MLM CE loss via AutoModelForMaskedLM, then average over up
    to --eval-iters random masks.

The two losses are NOT directly comparable across model types (BERT MLM
CE vs GPT-2 CLM CE), but they are comparable across subsets within the
same model type — which is what matters for subset comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Repo must be importable so we can load the nanoGPT GPT class.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


ALL_SUBSETS = [
    "mammal_centered",
    *[f"eukaryote_matched_random_seed{i}" for i in range(1, 11)],
    *[f"global_random_seed{i}" for i in range(1, 11)],
]


def ckpt_paths(base_dir: Path, subset: str, model_type: str) -> Dict[str, Path]:
    if model_type == "bert":
        ckpt = (
            base_dir / "genome_sequence" / "bert-output"
            / f"genome_sequence-small-{subset}" / "checkpoint-60000"
        )
        arrow = base_dir / "genome_sequence" / subset / "training_ready_hf_dataset_bert"
        return {"ckpt": ckpt, "arrow": arrow}
    elif model_type == "gpt2":
        ckpt = (
            base_dir / "genome_sequence" / "gpt2-output"
            / f"genome_sequence-small-{subset}" / "checkpoint-50000"
            / "training_state.bin"
        )
        arrow = base_dir / "genome_sequence" / subset / "training_ready_hf_dataset_gpt2"
        return {"ckpt": ckpt, "arrow": arrow}
    raise ValueError(f"unknown model_type={model_type}")


def eval_bert(ckpt_dir: Path, arrow_dir: Path, *, device: str, eval_iters: int,
              mlm_probability: float, seed: int,
              tokenizer_dir: Optional[Path] = None) -> Optional[Dict[str, float]]:
    import torch
    import numpy as np
    from datasets import load_from_disk
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    # Subset BERT ckpts ship only model weights / configs — the tokenizer
    # lives at <genome_sequence>/custom_tokenizer_bert_single_nuc/ shared
    # by every subset (single-nucleotide vocab=10).
    tok_path = tokenizer_dir if tokenizer_dir is not None else ckpt_dir
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path))
    model = AutoModelForMaskedLM.from_pretrained(str(ckpt_dir))
    model.to(device).eval()
    mask_id = tokenizer.mask_token_id
    special_ids = set(tokenizer.all_special_ids or ())

    ds = load_from_disk(str(arrow_dir))["valid"]
    if "input_ids" not in ds.column_names:
        return None
    n_rows = len(ds)
    if n_rows == 0:
        return None
    rng = np.random.default_rng(seed)

    losses = []
    with torch.no_grad():
        for _ in range(min(eval_iters, n_rows)):
            row = ds[int(rng.integers(0, n_rows))]
            ids = list(row["input_ids"])
            if not ids:
                continue
            base = torch.tensor(ids, dtype=torch.long, device=device)
            # Dynamic 15 % mask (training-equivalent).
            mask_prob = torch.full(base.shape, mlm_probability, device=device)
            # Don't mask special tokens.
            for sid in special_ids:
                mask_prob[base == sid] = 0.0
            masked = torch.bernoulli(mask_prob).bool()
            if masked.sum() == 0:
                continue
            input_ids = base.clone()
            input_ids[masked] = mask_id
            labels = torch.full_like(base, fill_value=-100)
            labels[masked] = base[masked]
            out = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=torch.ones_like(input_ids).unsqueeze(0),
                labels=labels.unsqueeze(0),
            )
            losses.append(float(out.loss.item()))

    if not losses:
        return None
    val_loss = float(sum(losses) / len(losses))
    return {"val_loss": val_loss, "n_batches": len(losses)}


def eval_gpt2(ckpt_file: Path, arrow_dir: Path, *, device: str,
              eval_iters: int, seed: int) -> Optional[Dict[str, float]]:
    import torch
    import numpy as np
    from datasets import load_from_disk
    from molcrawl.models.gpt2.model import GPT, GPTConfig

    state = torch.load(str(ckpt_file), map_location=device)
    model_args = state["model_args"]
    cfg = GPTConfig(**model_args)
    model = GPT(cfg)
    sd = state["model"]
    # Strip the nanoGPT torch.compile prefix if present.
    unwanted_prefix = "_orig_mod."
    for k in list(sd.keys()):
        if k.startswith(unwanted_prefix):
            sd[k[len(unwanted_prefix):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(device).eval()
    block_size = int(getattr(cfg, "block_size", 1024))

    ds = load_from_disk(str(arrow_dir))["valid"]
    if "input_ids" not in ds.column_names:
        return None
    n_rows = len(ds)
    if n_rows == 0:
        return None
    rng = np.random.default_rng(seed)

    losses = []
    with torch.no_grad():
        for _ in range(min(eval_iters, n_rows)):
            row = ds[int(rng.integers(0, n_rows))]
            ids = list(row["input_ids"])[:block_size]
            if len(ids) < 2:
                continue
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            y = x.clone()
            # nanoGPT trick: shift via y; model handles target_tokens internally.
            _, loss = model(x[:, :-1], y[:, 1:])
            losses.append(float(loss.item()))

    if not losses:
        return None
    val_loss = float(sum(losses) / len(losses))
    return {"val_loss": val_loss, "n_batches": len(losses)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-dir", default=os.environ.get("LEARNING_SOURCE_DIR"),
        help="Defaults to $LEARNING_SOURCE_DIR.",
    )
    ap.add_argument("--subsets", default=",".join(ALL_SUBSETS))
    ap.add_argument("--models", default="bert,gpt2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-iters", type=int, default=200)
    ap.add_argument("--mlm-probability", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.base_dir:
        ap.error("--base-dir is required (or set LEARNING_SOURCE_DIR in env).")
    base_dir = Path(args.base_dir)
    subsets = [s for s in args.subsets.split(",") if s]
    models = [m for m in args.models.split(",") if m]

    out_root = base_dir / "genome_sequence" / "analysis" / "val_loss_phase4_fixed"
    out_root.mkdir(parents=True, exist_ok=True)
    per_run_root = out_root / "per_run"
    per_run_root.mkdir(parents=True, exist_ok=True)

    print(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  base_dir   = {base_dir}")
    print(f"  subsets    = {len(subsets)}")
    print(f"  models     = {models}")
    print(f"  eval_iters = {args.eval_iters}")
    print(f"  device     = {args.device}")
    print(f"  out_root   = {out_root}")
    print()

    results: List[dict] = []
    t_total = time.time()
    for s in subsets:
        for m in models:
            paths = ckpt_paths(base_dir, s, m)
            ckpt, arrow = paths["ckpt"], paths["arrow"]
            tag = f"{m}__{s}"
            if args.dry_run:
                print(f"DRY: {tag}  ckpt={ckpt}  arrow={arrow}")
                continue
            if not ckpt.exists():
                print(f"  ✗ {tag}: ckpt missing → {ckpt}")
                results.append({
                    "subset": s, "model": m, "n_batches": 0,
                    "val_loss": None, "val_ppl": None, "status": "MISSING_CKPT",
                })
                continue
            if not arrow.exists():
                print(f"  ✗ {tag}: arrow missing → {arrow}")
                results.append({
                    "subset": s, "model": m, "n_batches": 0,
                    "val_loss": None, "val_ppl": None, "status": "MISSING_ARROW",
                })
                continue
            t0 = time.time()
            print(f"  RUN {tag} …", flush=True)
            try:
                if m == "bert":
                    tokenizer_dir = (
                        base_dir / "genome_sequence" / "custom_tokenizer_bert_single_nuc"
                    )
                    metrics = eval_bert(
                        ckpt, arrow, device=args.device,
                        eval_iters=args.eval_iters,
                        mlm_probability=args.mlm_probability,
                        seed=args.seed,
                        tokenizer_dir=tokenizer_dir,
                    )
                else:
                    metrics = eval_gpt2(
                        ckpt, arrow, device=args.device,
                        eval_iters=args.eval_iters,
                        seed=args.seed,
                    )
            except Exception as exc:
                elapsed = (time.time() - t0) / 60.0
                print(f"  ✗ {tag} ({elapsed:.1f} min): {exc}", flush=True)
                results.append({
                    "subset": s, "model": m, "n_batches": 0,
                    "val_loss": None, "val_ppl": None,
                    "status": f"FAILED: {exc}",
                })
                continue
            elapsed = (time.time() - t0) / 60.0
            if metrics is None:
                print(f"  ✗ {tag} ({elapsed:.1f} min): no batches")
                results.append({
                    "subset": s, "model": m, "n_batches": 0,
                    "val_loss": None, "val_ppl": None, "status": "EMPTY",
                })
                continue
            import math
            val_ppl = math.exp(min(metrics["val_loss"], 50.0))
            results.append({
                "subset": s, "model": m,
                "n_batches": metrics["n_batches"],
                "val_loss": metrics["val_loss"],
                "val_ppl": val_ppl,
                "status": "OK",
            })
            print(
                f"  ✓ {tag} ({elapsed:.1f} min): "
                f"val_loss={metrics['val_loss']:.4f}  ppl={val_ppl:.2f}  "
                f"(n_batches={metrics['n_batches']})",
                flush=True,
            )

            # Per-run log for trace.
            run_dir = per_run_root / tag
            run_dir.mkdir(exist_ok=True, parents=True)
            with (run_dir / "result.json").open("w") as f:
                json.dump({
                    "subset": s, "model": m, "ckpt": str(ckpt),
                    "arrow": str(arrow), **metrics,
                    "val_ppl": val_ppl,
                    "elapsed_min": elapsed,
                }, f, indent=2)

    # Aggregate CSV
    csv_path = out_root / "val_loss_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["subset", "model", "n_batches", "val_loss", "val_ppl", "status"],
        )
        w.writeheader()
        for r in results:
            w.writerow(r)

    # Human report
    txt_path = out_root / "val_loss_report.txt"
    lines = [
        "val_loss / val_perplexity — Phase 4 修正後の Arrow valid split で再計算",
        "=" * 78,
        f"generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"eval_iters:   {args.eval_iters}",
        "",
        f"{'subset':<35} {'BERT loss':>10} {'BERT ppl':>10} {'GPT2 loss':>10} {'GPT2 ppl':>10}",
        "-" * 78,
    ]
    by_subset: Dict[str, Dict[str, dict]] = {}
    for r in results:
        by_subset.setdefault(r["subset"], {})[r["model"]] = r
    for s in subsets:
        b = by_subset.get(s, {}).get("bert", {})
        g = by_subset.get(s, {}).get("gpt2", {})
        def fmt(d, key, w=10):
            v = d.get(key)
            return f"{v:>{w}.4f}" if isinstance(v, float) else f"{'-':>{w}}"
        lines.append(
            f"{s:<35} {fmt(b,'val_loss')} {fmt(b,'val_ppl')} "
            f"{fmt(g,'val_loss')} {fmt(g,'val_ppl')}"
        )
    txt_path.write_text("\n".join(lines) + "\n")

    print()
    total_min = (time.time() - t_total) / 60.0
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_fail = sum(1 for r in results if r["status"] != "OK")
    print(f"[done] total {total_min:.1f} min   ok={n_ok}  failed/skipped={n_fail}")
    print(f"  results: {csv_path}")
    print(f"  report : {txt_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
