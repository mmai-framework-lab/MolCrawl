"""Held-out perplexity for the compounds GPT-2 ladder.

The compounds plan asks for MoleculeNet plus "test split perplexity". The ladder
selected checkpoints on valid, so test never fed model selection and is reportable
as held out.

Scoring mirrors what training did — the same causal shift over the same packed
1024-token blocks — so the number sits on the same scale as the val losses in the
run logs. Perplexity is exp of the mean token loss.

Usage:
    LEARNING_SOURCE_DIR=<dir> python scripts/eval_gpt2_perplexity.py \
        --sizes small medium large ex-large --split test
"""
from __future__ import annotations

import json
import logging
import math
import os
from argparse import ArgumentParser
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gpt2_perplexity")

BATCH_BLOCKS = 16


def load_model(ckpt_path: Path, device: str):
    from molcrawl.models.gpt2.model import GPT, GPTConfig

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["model_args"]
    model = GPT(GPTConfig(**args))
    state = ckpt["model"]
    # torch.compile / DDP wrappers leave their prefix on the keys.
    for prefix in ("_orig_mod.", "module."):
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt, args


@torch.no_grad()
def score_split(model, dataset, device: str, max_blocks: int = 0) -> tuple[float, int]:
    """Mean token loss over the split, plus the number of scored tokens."""
    n = len(dataset) if not max_blocks else min(len(dataset), max_blocks)
    total_loss, total_tokens = 0.0, 0
    for start in range(0, n, BATCH_BLOCKS):
        stop = min(start + BATCH_BLOCKS, n)
        block = torch.tensor(dataset[start:stop]["input_ids"], dtype=torch.long, device=device)
        # GPT.forward calls targets.view(-1), which rejects a non-contiguous slice.
        # Training never hit this because get_batch pins the batch first, and
        # pin_memory returns a contiguous copy.
        x, y = block[:, :-1].contiguous(), block[:, 1:].contiguous()
        _, loss = model(x, y)
        tokens = y.numel()
        total_loss += float(loss) * tokens
        total_tokens += tokens
        if start and start % (BATCH_BLOCKS * 200) == 0:
            logger.info("    %d/%d blocks, running loss %.4f", stop, n, total_loss / total_tokens)
    return total_loss / total_tokens, total_tokens


def main() -> int:
    parser = ArgumentParser(description="Held-out perplexity for the compounds GPT-2 ladder")
    parser.add_argument("--sizes", nargs="+", default=["small", "medium", "large", "ex-large"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-blocks", type=int, default=0, help="0 scores the whole split")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.environ.get("LEARNING_SOURCE_DIR"):
        raise SystemExit("LEARNING_SOURCE_DIR is not set")

    from datasets import load_from_disk

    from molcrawl.core.paths import COMPOUNDS_DATASET_DIR_GPT2, get_gpt2_output_path

    dataset = load_from_disk(COMPOUNDS_DATASET_DIR_GPT2)[args.split]
    logger.info("%s split: %d blocks", args.split, len(dataset))

    results = {}
    for size in args.sizes:
        ckpt_path = Path(get_gpt2_output_path("compounds", size)) / "ckpt.pt"
        if not ckpt_path.exists():
            logger.warning("%s: no checkpoint at %s, skipping", size, ckpt_path)
            continue
        logger.info("%s: loading %s", size, ckpt_path)
        model, ckpt, model_args = load_model(ckpt_path, args.device)
        loss, tokens = score_split(model, dataset, args.device, args.max_blocks)
        results[size] = {
            "loss": loss,
            "perplexity": math.exp(loss),
            "tokens_scored": tokens,
            "checkpoint_iter": ckpt.get("iter_num"),
            "best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
            "params_millions": sum(p.numel() for p in model.parameters()) / 1e6,
        }
        logger.info(
            "%s: %s loss %.4f, perplexity %.3f (val %.4f at iter %s)",
            size, args.split, loss, results[size]["perplexity"],
            results[size]["best_val_loss"], results[size]["checkpoint_iter"],
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"\n{'size':10s} {'params(M)':>10s} {'val loss':>9s} {args.split + ' loss':>10s} {'perplexity':>11s}")
    for size, r in results.items():
        print(f"{size:10s} {r['params_millions']:>10.1f} {r['best_val_loss']:>9.4f} "
              f"{r['loss']:>10.4f} {r['perplexity']:>11.3f}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {"split": args.split, "blocks": len(dataset), "results": results}
        (out / f"{args.split}_perplexity.json").write_text(json.dumps(payload, indent=2))
        logger.info("wrote %s", out / f"{args.split}_perplexity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
