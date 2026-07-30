"""Compute MLM accuracy for one BERT checkpoint on its val/test split.

Implements Task B from collect_base_metrics_instructions.md. Designed
to be invoked one ckpt at a time via SLURM so multiple can run in
parallel.

Tokenizer resolution reuses the same 3-tier fallback as
``molcrawl.tasks.evaluation._adapters.hf_mlm_adapter.HfMlmAdapter``
(``--tokenizer-dir`` → ckpt dir → ``(arch, modality)`` built-in /
custom_tokenizer_<arch>/). This is the only sane way to handle the
4-5 different per-modality tokenizer styles in this repo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def find_val_split(val_dir: Path) -> Path | None:
    for candidate in ("valid", "validation", "test"):
        p = val_dir / candidate
        if p.exists() and (p / "dataset_info.json").exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--val-dir", required=True,
                    help="training_ready_hf_dataset/ root (contains train/valid/test)")
    ap.add_argument("--tokenizer-dir", default=None,
                    help="Override tokenizer dir; otherwise resolved via HfMlmAdapter "
                         "fallback chain (modality + arch).")
    ap.add_argument("--modality", required=True,
                    help="Modality string passed to HfMlmAdapter fallback "
                         "(protein_sequence / rna / compounds / molecule_nat_lang / "
                         "genome_sequence)")
    ap.add_argument("--arch", default="bert",
                    help="Architecture string (default bert)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlm-probability", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Repo on sys.path so we can import the adapter machinery
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT))

    ckpt_dir = Path(args.ckpt_dir)
    val_root = Path(args.val_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (ckpt_dir / "config.json").exists():
        print(f"[ERROR] no config.json in {ckpt_dir}", file=sys.stderr)
        return 1
    split = find_val_split(val_root)
    if split is None:
        print(f"[ERROR] no valid/validation/test split under {val_root}", file=sys.stderr)
        return 1

    import torch
    import numpy as np
    from datasets import load_from_disk
    from transformers import DataCollatorForLanguageModeling
    from torch.utils.data import DataLoader
    from molcrawl.tasks.evaluation._base.model_adapter import ModelHandle
    from molcrawl.tasks.evaluation._adapters.hf_mlm_adapter import HfMlmAdapter

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use the adapter's 3-tier tokenizer resolution + model load
    handle = ModelHandle(
        arch=args.arch,
        modality=args.modality,
        model_path=str(ckpt_dir),
        tokenizer_path=args.tokenizer_dir,
        size=None,
        extras={"device": device},
    )
    adapter = HfMlmAdapter(handle)
    print(f"[load] HfMlmAdapter for {args.label} ({args.arch}/{args.modality})", flush=True)
    adapter.load()
    tokenizer = adapter.tokenizer
    model = adapter.model
    model.eval()
    print(f"  tokenizer={type(tokenizer).__name__}, vocab={tokenizer.vocab_size}", flush=True)

    print(f"[load] dataset from {split}", flush=True)
    dataset = load_from_disk(str(split))
    if len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))
    print(f"  using {len(dataset)} samples (split={split.name})", flush=True)

    # The dataset may carry pre-tokenised input_ids already; in that case
    # DataCollatorForLanguageModeling wants pad_token_id set.
    if tokenizer.pad_token_id is None:
        if tokenizer.mask_token_id is not None:
            tokenizer.pad_token = tokenizer.mask_token
        else:
            tokenizer.pad_token = "[PAD]"
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collator,
        shuffle=False, num_workers=2,
    )

    total_correct = 0
    total_masked = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = out.logits.argmax(dim=-1)
            mask = labels != -100
            correct = ((preds == labels) & mask).sum().item()
            total_correct += correct
            total_masked  += mask.sum().item()

    acc = total_correct / total_masked if total_masked > 0 else None
    elapsed_min = (time.time() - t0) / 60.0
    rec = {
        "label": args.label,
        "modality": args.modality,
        "arch": args.arch,
        "ckpt_dir": str(ckpt_dir),
        "val_dir": str(val_root),
        "split_used": split.name,
        "n_samples": int(len(dataset)),
        "n_masked": int(total_masked),
        "n_correct": int(total_correct),
        "mlm_accuracy": float(acc) if acc is not None else None,
        "mlm_probability": args.mlm_probability,
        "batch_size": args.batch_size,
        "elapsed_min": elapsed_min,
    }
    out_path = out_dir / "mlm_accuracy.json"
    out_path.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"\n[done] {args.label}: mlm_accuracy = "
          f"{rec['mlm_accuracy']:.4f} (n_masked={total_masked:,}, {elapsed_min:.1f} min)")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
