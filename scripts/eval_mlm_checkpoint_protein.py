"""Re-score a saved protein-BERT MLM checkpoint on the eval subset (protein tokenizer).

The shared scripts/eval_mlm_checkpoint.py is hardcoded to CompoundsTokenizer; this is
the protein-tokenizer twin. Its purpose: get the CORRECT [MASK]-only loss after the
metric fix (PR #141). The old training-time diagnostic omitted position_ids; for a
document_masking run (which trains with per-document reset position_ids) that mismatch
inflated eval_loss_mask and could make a learning run look collapsed. Pass
--drop-position-ids to reproduce the old (defective) number for comparison.
"""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_mlm_protein")
BATCH_ROWS = 16


def main() -> int:
    p = ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", default=None)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--document-masking", action="store_true")
    p.add_argument("--drop-position-ids", action="store_true", help="reproduce old defective diagnostic")
    p.add_argument("--mlm-probability", type=float, default=0.2)
    p.add_argument("--label", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from datasets import load_from_disk
    from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling

    from molcrawl.data.protein_sequence.utils.bert_tokenizer import create_bert_protein_tokenizer
    from molcrawl.models.bert._mlm_diagnostics import split_mlm_loss
    from molcrawl.models.bert.main import resolve_eval_split, subsample_eval_split

    tok = create_bert_protein_tokenizer()
    dataset = load_from_disk(args.dataset_dir)
    split = args.split or resolve_eval_split(dataset.keys())
    rows = subsample_eval_split(dataset[split], random_sample=True)
    logger.info(
        "%s: %d rows from %s [%s] docmask=%s drop_pos=%s",
        args.label or "ckpt",
        len(rows),
        args.dataset_dir,
        split,
        args.document_masking,
        args.drop_position_ids,
    )

    base = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=args.mlm_probability)
    collator = base
    if args.document_masking:
        from molcrawl.models._collators import DocumentMaskingCollator

        collator = DocumentMaskingCollator(base, separator_id=tok.sep_token_id)
        logger.info("document masking on (separator id %s)", tok.sep_token_id)

    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint).to(args.device).eval()
    mask_id = tok.mask_token_id
    torch.manual_seed(0)

    sums = {k: 0.0 for k in ("mask", "copy", "random")}
    counts = {k: 0 for k in ("mask", "copy", "random")}
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_ROWS):
            feats = [{"input_ids": rows[i]["input_ids"]} for i in range(start, min(start + BATCH_ROWS, len(rows)))]
            batch = collator(feats)
            batch = {k: v.to(args.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            if args.drop_position_ids:
                batch.pop("position_ids", None)
            logits = model(**batch).logits
            s, c = split_mlm_loss(logits, labels, batch["input_ids"], mask_id)
            for k in sums:
                sums[k] += s[k]
                counts[k] += c[k]

    mask_loss = sums["mask"] / max(counts["mask"], 1)
    blended = sum(sums.values()) / max(sum(counts.values()), 1)
    out = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "split": split,
        "document_masking": args.document_masking,
        "drop_position_ids": args.drop_position_ids,
        "eval_loss_mask": round(mask_loss, 4),
        "eval_loss_blended": round(blended, 4),
        "unigram_baseline": 2.8947,
    }
    print("RESULT " + json.dumps(out))
    logger.info("[%s] eval_loss_mask=%.4f blended=%.4f (unigram 2.8947)", args.label or "ckpt", mask_loss, blended)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
