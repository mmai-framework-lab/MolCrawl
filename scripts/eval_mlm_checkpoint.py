"""Score a saved MLM checkpoint on the eval subset, without retraining.

The compounds BERT endpoints were all measured on the *first* 10,000 rows of their
eval split, which the source-ordered compounds sets make unrepresentative. The subset
is drawn at random now, so the endpoints that decided the production form have to be
re-measured on the new slice before they can be compared with anything.

This runs one forward pass per batch with the same collator training used, and reports
the loss split by position type - `[MASK]`, `copy`, `random` - because only the
`[MASK]` column is comparable to the MLM baselines (unigram 2.476, degenerate 2.221,
left-neighbour 1.932, both-neighbour 1.380).

Usage:
    LEARNING_SOURCE_DIR=<dir> python scripts/eval_mlm_checkpoint.py \
        --checkpoint <run>/checkpoint-15000 --dataset-dir <training_ready_bert> \
        --max-length 128 --label V1L
"""
from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_mlm")

BATCH_ROWS = 64


def main() -> int:
    parser = ArgumentParser(description="Re-score a saved MLM checkpoint on the eval subset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default=None, help="defaults to valid, else test")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--document-masking", action="store_true", help="match the arm's setting")
    parser.add_argument("--mlm-probability", type=float, default=0.2)
    parser.add_argument("--vocab", default="assets/molecules/vocab.txt")
    parser.add_argument(
        "--eval-subset",
        choices=("random", "head"),
        default="random",
        help="random reproduces the new slice; head reproduces the number a run reported before it",
    )
    parser.add_argument(
        "--drop-position-ids",
        action="store_true",
        help="reproduce the diagnostic forward in _mlm_diagnostics, which omits position_ids",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from datasets import load_from_disk
    from transformers import AutoModelForMaskedLM

    from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer
    from molcrawl.models._collators import make_mlm_collator
    from molcrawl.models.bert._mlm_diagnostics import split_mlm_loss
    from molcrawl.models.bert.main import resolve_eval_split, subsample_eval_split

    dataset = load_from_disk(args.dataset_dir)
    split = args.split or resolve_eval_split(dataset.keys())
    rows = subsample_eval_split(dataset[split], random_sample=args.eval_subset == "random")
    logger.info(
        "%s: %d rows from %s [%s], %s slice",
        args.label or "checkpoint", len(rows), args.dataset_dir, split, args.eval_subset,
    )

    tokenizer = CompoundsTokenizer(args.vocab, args.max_length)
    actual = getattr(tokenizer, "tokenizer", tokenizer)
    collator = make_mlm_collator(actual, ambiguous_tokens=[], mlm_probability=args.mlm_probability)
    if args.document_masking:
        from molcrawl.models._collators import DocumentMaskingCollator

        collator = DocumentMaskingCollator(collator, separator_id=actual.sep_token_id)
        logger.info("document masking on")

    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint).to(args.device).eval()
    mask_id = actual.mask_token_id

    # The collator draws the mask pattern at random. Seed it so a re-run of this script
    # reports the same number - the point of the exercise is a stable endpoint.
    torch.manual_seed(0)

    sums = {k: 0.0 for k in ("mask", "copy", "random")}
    counts = {k: 0 for k in ("mask", "copy", "random")}
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_ROWS):
            batch = collator([rows[i] for i in range(start, min(start + BATCH_ROWS, len(rows)))])
            batch = {k: v.to(args.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            if args.drop_position_ids:
                batch.pop("position_ids", None)
            logits = model(**batch).logits
            s, c = split_mlm_loss(logits, labels, batch["input_ids"], mask_id)
            for k in sums:
                sums[k] += s[k]
                counts[k] += c[k]
            if start and start % (BATCH_ROWS * 40) == 0:
                logger.info("    %d/%d rows, [MASK] loss %.4f", start, len(rows), sums["mask"] / max(counts["mask"], 1))

    result = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "dataset_dir": args.dataset_dir,
        "split": split,
        "eval_subset": args.eval_subset,
        "rows": len(rows),
        "document_masking": args.document_masking,
        "mlm_probability": args.mlm_probability,
    }
    for k in sums:
        result[f"loss_{k}"] = sums[k] / counts[k] if counts[k] else None
        result[f"count_{k}"] = counts[k]

    print(f"\n{args.label or 'checkpoint'} [{args.eval_subset} slice]: [MASK] {result['loss_mask']:.4f}  "
          f"copy {result['loss_copy']:.4f}  random {result['loss_random']:.4f}  "
          f"({result['count_mask']:,} masked targets)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
