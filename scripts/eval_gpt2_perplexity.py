"""Deterministic held-out loss / perplexity for a GPT-2 size ladder.

Two reasons to score a split here rather than read the number off the training log:

* **Deterministic and complete.** ``train.py`` estimates val loss from
  ``eval_iters`` batches drawn *with replacement* (``get_batch``), so every eval
  point carries sampling noise, and ``best_val`` — a minimum over ~32 such points —
  sits below the value it estimates by an amount that grows with that noise.
  ``eval_sequences`` equalises how many sequences each ladder size averages over,
  but the noise and the minimum-selection bias remain. How many points that minimum
  is taken over is per modality — ~31 for compounds, ~11 for protein. This script walks the whole
  split once, so neither applies.
* **test stays held out.** Checkpoints are selected on valid, so scoring test
  here is the first time test is touched.

Scoring mirrors training: the same causal shift over the same blocks, **and the same
targets excluded from the loss** — ``get_batch`` blanks ambiguous tokens (protein X B
Z, genome N and the IUPAC codes) and, where the config sets it, pad positions, and
``GPT.forward`` passes ``ignore_index``. Scoring without those exclusions would put
protein and genome on a different scale from their own training logs, so this script
refuses to run rather than quietly produce such a number. Perplexity is exp of the
mean loss over the tokens that are scored.

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

# RNABinDataset needs the same Geneformer token dictionary the RNA configs point at
# (configs/rna/gpt2_*.py). It raises on None, so leaving this unset would turn the
# documented invocation into a bare ValueError.
DEFAULT_RNA_VOCAB = Path(__file__).resolve().parents[1] / (
    "molcrawl/data/rna/dataset/geneformer/token_dictionary.pkl"
)


def checkpoint_path_for(modality: str, size: str, template: str | None) -> Path:
    """Where a ladder wrote its checkpoint for one size.

    Without a template this goes through get_gpt2_output_path, which normalises "xl"
    to "ex-large"; a template takes the size verbatim, which is what the protein
    ladder needs since it wrote runs/ladder-gpt2-xl.
    """
    from molcrawl.core.paths import get_gpt2_output_path

    if template:
        return Path(template.format(size=size))
    return Path(get_gpt2_output_path(modality, size)) / "ckpt.pt"


def check_modality_matches(ckpt, modality: str, ckpt_path: Path) -> str | None:
    """Compare the modality the checkpoint was trained on with the one being scored.

    train.py stores the resolved config, whose ``dataset`` key is the modality string,
    so a checkpoint scored against another modality's split can be caught rather than
    producing a plausible-looking number. Matters most with --checkpoint-template,
    where the path carries no modality.
    """
    trained_on = (ckpt.get("config") or {}).get("dataset")
    if trained_on and trained_on != modality:
        raise SystemExit(
            f"{ckpt_path} was trained on {trained_on!r} but --modality is {modality!r}; "
            "scoring it against this split would compare unrelated things. Pass the "
            "matching --modality, or --allow-modality-mismatch if this is deliberate."
        )
    return trained_on


def build_tokenizer(modality: str, genome_tokenizer_model: str | None):
    """The tokenizer whose ids the modality's ambiguous-token list refers to.

    Only needed where that list is non-empty, i.e. protein and genome.
    """
    if modality == "protein_sequence":
        from molcrawl.data.protein_sequence.dataset.tokenizer import EsmSequenceTokenizer

        return EsmSequenceTokenizer()
    if modality == "genome_sequence":
        import sentencepiece as spm

        from molcrawl.core.paths import get_refseq_tokenizer_path

        model_file = genome_tokenizer_model or get_refseq_tokenizer_path()
        if not Path(model_file).exists():
            raise SystemExit(
                f"genome tokenizer model not found at {model_file}; pass "
                "--genome-tokenizer-model. It is needed to blank N and the IUPAC "
                "codes from the loss, exactly as training does."
            )
        return spm.SentencePieceProcessor(model_file=str(model_file))
    return None


def resolve_ignored_target_ids(modality: str, genome_tokenizer_model: str | None, pad_token_id: int | None):
    """Target ids training excludes from the CLM loss, so scoring can exclude them too.

    get_batch blanks ambiguous tokens and, when the config sets it, the pad id;
    GPT.forward then passes ignore_index. Scoring without the same exclusions would
    not be on the scale of the run's own val losses.
    """
    from molcrawl.models._collators import (
        ambiguous_tokens_for_modality,
        resolve_ambiguous_token_ids,
    )

    ids = []
    tokens = ambiguous_tokens_for_modality(modality)
    if tokens:
        tokenizer = build_tokenizer(modality, genome_tokenizer_model)
        if tokenizer is None:
            raise SystemExit(
                f"{modality} excludes {tokens} from the training loss, but this script "
                "has no tokenizer for it, so the number would not be comparable to the "
                "run's val losses."
            )
        ids = list(resolve_ambiguous_token_ids(tokenizer, tokens))
    if pad_token_id is not None:
        ids.append(int(pad_token_id))
    return ids


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
        if args.dataset_dir:
            raise SystemExit("--dataset-dir does not apply to --modality rna; use --rna-bin-dir")
        vocab_file = args.rna_vocab_file or DEFAULT_RNA_VOCAB
        if not Path(vocab_file).exists():
            # Checked here so it fails next to --rna-bin-dir rather than as a bare
            # ValueError from inside the dataset.
            raise SystemExit(f"RNA gene vocabulary not found at {vocab_file}; pass --rna-vocab-file")
        logger.info("rna vocabulary: %s", vocab_file)
        ds = RNABinDataset(bin_dir, split=args.split, vocab_file=str(vocab_file))

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
def score_split(model, fetch, n_rows: int, device: str, max_sequences: int = 0, ignored_ids=()) -> tuple[float, int, int]:
    """Mean loss over the scored targets, the count of them, and sequences read."""
    from molcrawl.models._collators import mask_ambiguous_targets_for_clm
    from molcrawl.models._collators.ambiguity_aware_collator import IGNORE_INDEX

    n = n_rows if not max_sequences else min(n_rows, max_sequences)
    total_loss, total_tokens = 0.0, 0
    for start in range(0, n, BATCH_BLOCKS):
        stop = min(start + BATCH_BLOCKS, n)
        block = fetch(start, stop).to(device)
        # GPT.forward calls targets.view(-1), which rejects a non-contiguous slice.
        # Training never hit this because get_batch pins the batch first, and
        # pin_memory returns a contiguous copy.
        x, y = block[:, :-1].contiguous(), block[:, 1:].contiguous()
        if ignored_ids:
            # Same helper get_batch uses, so the excluded positions match exactly.
            y = mask_ambiguous_targets_for_clm(y, ignored_ids).contiguous()
        _, loss = model(x, y)
        # cross_entropy averages over the kept targets only, so weight by those.
        tokens = int((y != IGNORE_INDEX).sum()) if ignored_ids else y.numel()
        if not tokens:
            continue
        total_loss += float(loss) * tokens
        total_tokens += tokens
        if start and start % (BATCH_BLOCKS * 200) == 0:
            logger.info("    %d/%d sequences, running loss %.4f", stop, n, total_loss / total_tokens)
    return total_loss / total_tokens, total_tokens, n


def main() -> int:
    parser = ArgumentParser(description="Deterministic held-out perplexity for a GPT-2 ladder")
    parser.add_argument("--modality", default="compounds", choices=MODALITIES)
    parser.add_argument("--sizes", nargs="+", default=["small", "medium", "large", "ex-large"])
    parser.add_argument("--split", default="test", help="valid or test")
    parser.add_argument("--dataset-dir", default=None, help="override the modality's dataset dir")
    parser.add_argument("--rna-bin-dir", default=None, help="dir holding <split>.bin/<split>.json")
    parser.add_argument(
        "--rna-vocab-file",
        default=None,
        help=f"Geneformer token dictionary; defaults to the one the configs use ({DEFAULT_RNA_VOCAB})",
    )
    parser.add_argument(
        "--genome-tokenizer-model",
        default=None,
        help="SentencePiece model for genome, needed to exclude N/IUPAC from the loss as training does",
    )
    parser.add_argument(
        "--pad-token-id-for-loss",
        type=int,
        default=None,
        help=(
            "Pass only when the run's config sets pad_token_id_for_loss, and pass that "
            "value. In this repo only configs/genome_sequence/gpt2_small_subset.py does "
            "(= 5); the GPT-2 ladders do not. Passing it otherwise excludes targets "
            "training scored, which is the scale mismatch this script exists to avoid"
        ),
    )
    parser.add_argument("--output-name", default=None, help="JSON filename; defaults to <modality>_<split>_perplexity.json")
    parser.add_argument(
        "--allow-modality-mismatch",
        action="store_true",
        help="Score a checkpoint whose config names a different modality than --modality",
    )
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
    parser.add_argument("--max-sequences", type=int, default=0, help="0 scores the whole split")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.environ.get("LEARNING_SOURCE_DIR"):
        raise SystemExit("LEARNING_SOURCE_DIR is not set")

    # Resolve checkpoints first: opening the split reads a multi-GB dataset, and with
    # nothing to score that work buys an empty table.
    checkpoints = {}
    for size in args.sizes:
        ckpt_path = checkpoint_path_for(args.modality, size, args.checkpoint_template)
        if ckpt_path.exists():
            checkpoints[size] = ckpt_path
        else:
            logger.warning("%s: no checkpoint at %s, skipping", size, ckpt_path)
    if not checkpoints:
        raise SystemExit(f"no checkpoints found for sizes {args.sizes}")

    ignored_ids = resolve_ignored_target_ids(
        args.modality, args.genome_tokenizer_model, args.pad_token_id_for_loss
    )
    logger.info(
        "targets excluded from the loss: %s%s",
        ignored_ids or "none",
        "" if ignored_ids else " (this modality excludes nothing during training either)",
    )

    fetch, n_rows = open_split(args)

    results = {}
    for size, ckpt_path in checkpoints.items():
        logger.info("%s: loading %s", size, ckpt_path)
        model, ckpt, model_args = load_model(ckpt_path, args.device)
        trained_on = None if args.allow_modality_mismatch else check_modality_matches(ckpt, args.modality, ckpt_path)
        loss, tokens, sequences_scored = score_split(
            model, fetch, n_rows, args.device, args.max_sequences, ignored_ids
        )
        results[size] = {
            "loss": loss,
            "perplexity": math.exp(loss),
            "checkpoint": str(ckpt_path),
            "trained_on": trained_on,
            "sequences_scored": sequences_scored,
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
            "sequences_in_split": n_rows,
            # Excluded from the loss, matching get_batch. Recorded because the number
            # is only comparable to a val loss produced with the same exclusions.
            "ignored_target_ids": ignored_ids,
            "results": results,
        }
        name = args.output_name or f"{args.modality}_{args.split}_perplexity.json"
        (out / name).write_text(json.dumps(payload, indent=2))
        logger.info("wrote %s", out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
