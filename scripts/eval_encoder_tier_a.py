"""Tier A perplexity / accuracy evaluation for encoder (MLM) checkpoints.

Companion to ``molcrawl/models/gpt2/test_checkpoint.py`` — same evaluation
philosophy but for the BERT / RoBERTa / ESM2 / DNABERT-2 / ChemBERTa-2
family. Given a HuggingFace MLM checkpoint and a
``training_ready_hf_dataset`` directory, measures:

* MLM cross-entropy loss on the ``valid`` (or ``test``) split, using
  standard 15 %-masking via
  :class:`transformers.DataCollatorForLanguageModeling` so the number
  matches what the Trainer reports during pretraining.
* Perplexity = exp(loss).
* Top-1 accuracy at the masked positions.

Usage::

    python scripts/eval_encoder_tier_a.py \\
        --checkpoint_path .../bert-output/protein_sequence-small/checkpoint-55000 \\
        --dataset_dir .../protein_sequence/training_ready_hf_dataset \\
        --domain protein_sequence \\
        --arch bert \\
        --max_test_samples 2000

The tokenizer is resolved via the same fallback chain as
``HfMlmAdapter`` (`molcrawl.tasks.evaluation._adapters.hf_mlm_adapter`):
``--tokenizer_path`` → checkpoint dir → arch/modality default.

Stdout reports the same ``perplexity:`` / ``Top-1 accuracy:`` markers as
``test_checkpoint.py`` so the existing Tier A summary scripts can grep
them without modification.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)

# Make the molcrawl package importable so we can reuse the adapter's tokenizer
# fallback. Run-from-repo-root is the conventional invocation; we still add
# the parent directory in case someone runs the script from elsewhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("eval_encoder_tier_a")


_TRUSTED_REMOTE_CODE_HINTS = (
    "DNABERT-2",
    "DNABERT2",
    "dnabert-2",
    "InstaDeepAI",
    "NucleotideTransformer",
)


def _trust_remote_code_for(path: str) -> bool:
    if os.environ.get("TRUST_REMOTE_CODE", "").strip() == "1":
        return True
    return any(hint in path for hint in _TRUSTED_REMOTE_CODE_HINTS)


def load_tokenizer(
    checkpoint_path: str,
    tokenizer_path: Optional[str],
    arch: str,
    modality: str,
) -> PreTrainedTokenizerBase:
    """Mirror HfMlmAdapter's tokenizer resolution (1) explicit → (2) ckpt dir → (3) arch/modality."""
    trust = _trust_remote_code_for(checkpoint_path) or _trust_remote_code_for(tokenizer_path or "")

    tried: List[str] = []
    if tokenizer_path:
        tried.append(tokenizer_path)
        try:
            return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AutoTokenizer(%s) failed: %s — falling back.", tokenizer_path, exc)

    tried.append(checkpoint_path)
    try:
        return AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=trust)
    except Exception as exc:  # noqa: BLE001
        logger.info("Checkpoint %s carries no tokenizer (%s); using arch+modality fallback.",
                    checkpoint_path, exc)

    # arch/modality fallback — mirror hf_mlm_adapter._arch_modality_fallback
    if arch in ("bert", "chemberta2") and modality == "compounds":
        from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer
        return CompoundsTokenizer("assets/molecules/vocab.txt")

    if arch == "bert" and modality == "molecule_nat_lang":
        from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer
        return MoleculeNatLangTokenizer()

    if arch in ("bert", "esm2") and modality == "protein_sequence":
        from molcrawl.data.protein_sequence.utils.bert_tokenizer import BertProteinSequenceTokenizer
        return BertProteinSequenceTokenizer()

    if (arch, modality) in (
        ("bert", "genome_sequence"),
        ("bert", "rna"),
        ("dnabert2", "genome_sequence"),
    ):
        from molcrawl.core.paths import get_custom_tokenizer_path

        tok_dir = get_custom_tokenizer_path(modality, arch)
        tried.append(tok_dir)
        return AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=trust)

    # RoBERTa pretrain — same per-modality tokenizers as BERT (they share assets).
    if arch == "roberta":
        if modality == "compounds":
            from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer
            return CompoundsTokenizer("assets/molecules/vocab.txt")
        if modality == "molecule_nat_lang":
            from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer
            return MoleculeNatLangTokenizer()
        if modality == "protein_sequence":
            from molcrawl.data.protein_sequence.utils.bert_tokenizer import BertProteinSequenceTokenizer
            return BertProteinSequenceTokenizer()
        if modality in ("genome_sequence", "rna"):
            from molcrawl.core.paths import get_custom_tokenizer_path

            # roberta reuses the bert custom tokenizer for these modalities
            tok_dir = get_custom_tokenizer_path(modality, "bert")
            tried.append(tok_dir)
            return AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=trust)

    raise ValueError(
        f"No tokenizer fallback for arch={arch!r} modality={modality!r}. "
        f"Tried {tried}. Pass --tokenizer_path."
    )


def evaluate_mlm(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Any,
    device: str,
    batch_size: int,
    mlm_probability: float,
    seed: int,
) -> dict:
    """Compute mean MLM CE / perplexity / Top-1 accuracy on a HF dataset.

    Uses ``DataCollatorForLanguageModeling`` so the masking schedule and label
    layout match training-time exactly. The dataset is expected to expose an
    ``input_ids`` column of equal-length lists (the ``training_ready_hf_dataset``
    layout); ``attention_mask`` is reconstructed from the pad token id.
    """
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    def _add_mask(batch):
        # The dataset may not carry attention_mask; compute it from pad ids.
        if "attention_mask" not in batch:
            batch["attention_mask"] = [
                [int(t != pad_id) for t in seq] for seq in batch["input_ids"]
            ]
        return batch

    # Apply lazily (no .map() write-back) by wrapping the collator.
    def _collate(features):
        for f in features:
            if "attention_mask" not in f:
                f["attention_mask"] = [int(t != pad_id) for t in f["input_ids"]]
        return collator(features)

    # Deterministic masking: seed torch and numpy before the loop so two runs
    # with the same seed yield identical metrics.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )

    total_loss = 0.0
    total_masked = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # outputs.loss is mean CE over masked positions
            loss = outputs.loss.item()

            # Count masked positions in this batch to weight the average.
            masked = (labels != -100).sum().item()
            total_loss += loss * masked
            total_masked += masked

            # Top-1 accuracy at masked positions.
            preds = outputs.logits.argmax(dim=-1)
            mask = labels != -100
            correct += (preds[mask] == labels[mask]).sum().item()

    if total_masked == 0:
        return {"mlm_loss": float("nan"), "perplexity": float("nan"), "top1_accuracy": float("nan"),
                "num_masked": 0}

    mean_loss = total_loss / total_masked
    ppl = float(np.exp(mean_loss)) if mean_loss < 50 else float("inf")
    top1 = correct / total_masked
    return {
        "mlm_loss": mean_loss,
        "perplexity": ppl,
        "top1_accuracy": top1,
        "num_masked": total_masked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tier A MLM evaluation for HF encoder checkpoints"
    )
    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to a checkpoint-NNNN directory (HF Trainer format).")
    parser.add_argument("--dataset_dir", required=True,
                        help="Path to a training_ready_hf_dataset directory.")
    parser.add_argument("--domain", required=True,
                        choices=("protein_sequence", "genome_sequence", "molecule_nat_lang",
                                 "compounds", "rna"),
                        help="Modality key for tokenizer fallback.")
    parser.add_argument("--arch", default="bert",
                        choices=("bert", "roberta", "esm2", "chemberta2", "dnabert2"),
                        help="Architecture for tokenizer fallback.")
    parser.add_argument("--split", default="valid", choices=("train", "valid", "test"),
                        help="Which split to evaluate on.")
    parser.add_argument("--max_test_samples", type=int, default=2000,
                        help="Cap on the number of samples to evaluate (0 = all).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Eval batch size.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Probability of masking each token (matches Trainer default).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the masking schedule.")
    parser.add_argument("--device", default="cuda",
                        help="cuda or cpu. Auto-fallback to cpu if cuda is unavailable.")
    parser.add_argument("--tokenizer_path", default=None,
                        help="Optional explicit tokenizer dir; otherwise resolved via fallback.")
    parser.add_argument("--output_dir", default=None,
                        help="If set, writes ``results.json`` and ``stdout`` snapshot here.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("--device=cuda but cuda not available; falling back to cpu.")
        args.device = "cpu"

    trust = _trust_remote_code_for(args.checkpoint_path) or _trust_remote_code_for(args.tokenizer_path or "")
    logger.info("Loading model from %s (trust_remote_code=%s)", args.checkpoint_path, trust)
    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint_path, trust_remote_code=trust)
    model.to(args.device)

    tokenizer = load_tokenizer(args.checkpoint_path, args.tokenizer_path, args.arch, args.domain)
    # DataCollatorForLanguageModeling needs mask_token_id and pad_token_id.
    if tokenizer.mask_token_id is None:
        raise RuntimeError("Tokenizer has no mask_token_id — cannot run MLM eval.")

    logger.info("Loading dataset split %r from %s", args.split, args.dataset_dir)
    ds = load_from_disk(args.dataset_dir)
    if hasattr(ds, "keys") and args.split in ds:
        split = ds[args.split]
    else:
        # Direct split dir layout (e.g. training_ready_hf_dataset/valid/).
        split = load_from_disk(os.path.join(args.dataset_dir, args.split))

    if args.max_test_samples and len(split) > args.max_test_samples:
        # Take the first N rows — deterministic; equivalent to gpt2 test_checkpoint.
        split = split.select(range(args.max_test_samples))
    logger.info("Evaluating on %d samples", len(split))

    metrics = evaluate_mlm(
        model=model,
        tokenizer=tokenizer,
        dataset=split,
        device=args.device,
        batch_size=args.batch_size,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
    )

    # Print stdout markers matching test_checkpoint.py so existing Tier A
    # summary grep patterns ("perplexity: <float>", "Top-1 accuracy: <float>")
    # work unchanged.
    print(f"checkpoint: {args.checkpoint_path}")
    print(f"split: {args.split}  samples: {len(split)}  masked_positions: {metrics['num_masked']}")
    print(f"mlm_loss: {metrics['mlm_loss']:.6f}")
    print(f"perplexity: {metrics['perplexity']:.4f}")
    print(f"Top-1 accuracy: {metrics['top1_accuracy']:.6f}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump({**metrics, "checkpoint": args.checkpoint_path, "split": args.split,
                       "samples": len(split), "arch": args.arch, "domain": args.domain}, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
