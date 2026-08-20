"""Deterministic held-out loss / perplexity for a GPT-2 size ladder.

Two reasons to score a split here rather than read the number off the training log:

* **Deterministic and complete.** ``train.py`` estimates val loss from
  ``eval_iters`` batches drawn *with replacement* (``get_batch``), and
  ``eval_iters`` counts batches, not sequences. A ladder that shrinks
  ``batch_size`` for the larger models therefore evaluates them on fewer
  sequences, and ``best_val`` — a minimum over eval points — picks up a
  selection bias that grows with that noise. This script walks the whole split
  once, so the sizes are comparable.
* **test stays held out.** Checkpoints are selected on valid, so scoring test
  here is the first time test is touched.

Scoring mirrors training: the same causal shift over the same blocks, so the
numbers sit on the same scale as the val losses in the run logs. Perplexity is
exp of the mean token loss.

Usage:
    LEARNING_SOURCE_DIR=<dir> python scripts/eval_gpt2_perplexity.py \
        --modality compounds --sizes small medium large ex-large --split test

    # RNA reads the flat uint16 memmap the training runs use
    LEARNING_SOURCE_DIR=<dir> python scripts/eval_gpt2_perplexity.py \
        --modality rna --rna-bin-dir <dir with valid.bin/valid.json> --split valid
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

# Modality -> the paths constant holding its GPT-2 training_ready dataset. RNA is
# absent on purpose: its runs read a flat memmap, so it needs --rna-bin-dir.
DATASET_DIR_CONSTANTS = {
    "compounds": "COMPOUNDS_DATASET_DIR_GPT2",
    "protein_sequence": "UNIPROT_DATASET_DIR",
    "molecule_nat_lang": "MOLECULE_NAT_LANG_DATASET_DIR",
    "genome_sequence": "REFSEQ_DATASET_DIR",
}
MODALITIES = sorted(set(DATASET_DIR_CONSTANTS) | {"rna"})


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


def open_split(args):
    """Return (fetch, n_rows) where fetch(start, stop) gives a (rows, block) int64 tensor."""
    if args.modality == "rna":
        from molcrawl.data.rna.dataset.rna_dataset import RNABinDataset

        bin_dir = args.rna_bin_dir or os.environ.get("RNA_BIN_DIR")
        if not bin_dir:
            raise SystemExit("--modality rna needs --rna-bin-dir (or RNA_BIN_DIR)")
        ds = RNABinDataset(bin_dir, split=args.split, vocab_file=args.rna_vocab_file)

        def fetch(start: int, stop: int):
            return torch.stack([ds[i] for i in range(start, stop)])

        return fetch, len(ds)

    from datasets import load_from_disk

    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        from molcrawl.core import paths

        dataset_dir = getattr(paths, DATASET_DIR_CONSTANTS[args.modality])
    ds = load_from_disk(dataset_dir)[args.split]
    logger.info("%s %s split: %d sequences from %s", args.modality, args.split, len(ds), dataset_dir)

    def fetch(start: int, stop: int):
        return torch.tensor(ds[start:stop]["input_ids"], dtype=torch.long)

    return fetch, len(ds)


@torch.no_grad()
def score_split(model, fetch, n_rows: int, device: str, max_blocks: int = 0) -> tuple[float, int]:
    """Mean token loss over the split, plus the number of scored tokens."""
    n = n_rows if not max_blocks else min(n_rows, max_blocks)
    total_loss, total_tokens = 0.0, 0
    for start in range(0, n, BATCH_BLOCKS):
        stop = min(start + BATCH_BLOCKS, n)
        block = fetch(start, stop).to(device)
        # GPT.forward calls targets.view(-1), which rejects a non-contiguous slice.
        # Training never hit this because get_batch pins the batch first, and
        # pin_memory returns a contiguous copy.
        x, y = block[:, :-1].contiguous(), block[:, 1:].contiguous()
        _, loss = model(x, y)
        tokens = y.numel()
        total_loss += float(loss) * tokens
        total_tokens += tokens
        if start and start % (BATCH_BLOCKS * 200) == 0:
            logger.info("    %d/%d sequences, running loss %.4f", stop, n, total_loss / total_tokens)
    return total_loss / total_tokens, total_tokens


def main() -> int:
    parser = ArgumentParser(description="Deterministic held-out perplexity for a GPT-2 ladder")
    parser.add_argument("--modality", default="compounds", choices=MODALITIES)
    parser.add_argument("--sizes", nargs="+", default=["small", "medium", "large", "ex-large"])
    parser.add_argument("--split", default="test", help="valid or test")
    parser.add_argument("--dataset-dir", default=None, help="override the modality's dataset dir")
    parser.add_argument("--rna-bin-dir", default=None, help="dir holding <split>.bin/<split>.json")
    parser.add_argument("--rna-vocab-file", default=None)
    parser.add_argument(
        "--checkpoint-template",
        default=None,
        help=(
            "Path pattern with a {size} placeholder, for ladders that do not sit at "
            "get_gpt2_output_path. The protein ladder is one: its runs wrote "
            ".../protein_sequence/runs/ladder-gpt2-{size}/ckpt.pt, and it uses 'xl' "
            "rather than the 'ex-large' the path helper normalises to."
        ),
    )
    parser.add_argument("--max-blocks", type=int, default=0, help="0 scores the whole split")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.environ.get("LEARNING_SOURCE_DIR"):
        raise SystemExit("LEARNING_SOURCE_DIR is not set")

    from molcrawl.core.paths import get_gpt2_output_path

    fetch, n_rows = open_split(args)

    results = {}
    for size in args.sizes:
        if args.checkpoint_template:
            ckpt_path = Path(args.checkpoint_template.format(size=size))
        else:
            ckpt_path = Path(get_gpt2_output_path(args.modality, size)) / "ckpt.pt"
        if not ckpt_path.exists():
            logger.warning("%s: no checkpoint at %s, skipping", size, ckpt_path)
            continue
        logger.info("%s: loading %s", size, ckpt_path)
        model, ckpt, model_args = load_model(ckpt_path, args.device)
        loss, tokens = score_split(model, fetch, n_rows, args.device, args.max_blocks)
        results[size] = {
            "loss": loss,
            "perplexity": math.exp(loss),
            "tokens_scored": tokens,
            "checkpoint_iter": ckpt.get("iter_num"),
            "best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
            "params_millions": sum(p.numel() for p in model.parameters()) / 1e6,
        }
        logger.info(
            "%s: %s loss %.4f, perplexity %.3f (logged best_val %.4f at iter %s)",
            size, args.split, loss, results[size]["perplexity"],
            results[size]["best_val_loss"], results[size]["checkpoint_iter"],
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    header = f"{'size':10s} {'params(M)':>10s} {'logged best_val':>16s} {args.split + ' loss':>12s} {'perplexity':>11s}"
    print(f"\n{header}")
    for size, r in results.items():
        print(f"{size:10s} {r['params_millions']:>10.1f} {r['best_val_loss']:>16.4f} "
              f"{r['loss']:>12.4f} {r['perplexity']:>11.3f}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "modality": args.modality,
            "split": args.split,
            "sequences": n_rows,
            "results": results,
        }
        (out / f"{args.modality}_{args.split}_perplexity.json").write_text(json.dumps(payload, indent=2))
        logger.info("wrote %s", out / f"{args.modality}_{args.split}_perplexity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
