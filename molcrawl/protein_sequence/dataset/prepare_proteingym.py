"""
ProteinGym Dataset Preparation Script for GPT-2 / BERT Fine-tuning

Converts ProteinGym v1.3 DMS substitution assay CSVs into the
training_ready_hf_dataset format compatible with the existing
protein_sequence training pipeline.

Pipeline:
    1. Read every ``<assay>.csv`` from the substitutions directory.
    2. Collect ``mutated_sequence`` (mutant variants) AND ``target_seq``
       (wild-type) as training sequences.
    3. Deduplicate by exact sequence string.
    4. Shuffle and apply an 80 / 10 / 10 train / valid / test split.
    5. Tokenise with ``EsmSequenceTokenizer`` (character-level, vocab_size=33).
    6. Concatenate all token sequences into one long stream (with EOS between
       sequences), then chunk into fixed-length blocks of *context_length*.
    7. Save as a HuggingFace DatasetDict to ``output_dir/training_ready_hf_dataset/``.

Usage (standalone):
    LEARNING_SOURCE_DIR=learning_source_20260311 \\
    python -m molcrawl.protein_sequence.dataset.prepare_proteingym \\
        assets/configs/protein_sequence.yaml

    The output is saved to:
        $LEARNING_SOURCE_DIR/protein_sequence/proteingym/training_ready_hf_dataset/
"""

import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from datasets import Dataset, DatasetDict

from molcrawl.protein_sequence.dataset.tokenizer import EsmSequenceTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-level helpers (top-level so they can be used with num_proc > 1)
# ---------------------------------------------------------------------------


def _tokenize_sequence(sequence: str, tokenizer: EsmSequenceTokenizer) -> List[int]:
    """Return a list of integer token IDs for one protein sequence string."""
    encoded = tokenizer(sequence, add_special_tokens=True)
    return encoded["input_ids"]


def _concatenate_texts(
    examples: Dict,
    eos_token_id: int,
) -> Dict:
    """
    Concatenate all ``input_ids`` lists in a batch into one flat list,
    inserting *eos_token_id* between sequences.
    """
    all_ids: List[int] = []
    for ids in examples["input_ids"]:
        all_ids.extend(ids)
        all_ids.append(eos_token_id)
    return {"input_ids": all_ids}


def _create_chunks(examples: Dict, context_length: int) -> Dict:
    """Split a flat ``input_ids`` list into fixed-length blocks."""
    ids = examples["input_ids"]
    n_chunks = len(ids) // context_length
    chunks = [ids[i * context_length : (i + 1) * context_length] for i in range(n_chunks)]
    return {"input_ids": chunks}


# ---------------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------------


def prepare_proteingym(
    source_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    context_length: int = 1024,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_proc: int = 4,
) -> str:
    """
    Load ProteinGym DMS substitution CSV files from *source_dir*, build a
    language-model training dataset, and save it to *output_dir*.

    Args:
        source_dir: Directory containing individual assay ``*.csv`` files
                    (produced by ``download_proteingym()``).
        output_dir: Parent directory; the dataset is written under
                    ``output_dir/training_ready_hf_dataset/``.
        context_length: Token block length (default 1024).
        train_ratio: Fraction of sequences used for training (rest split 50/50
                     between validation and test).
        seed: Random seed for reproducible shuffling.
        num_proc: Parallel workers for HuggingFace Dataset.map() operations.

    Returns:
        Absolute path string of the saved dataset directory.
    """
    import pandas as pd

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(
            f"ProteinGym source directory not found: {source_dir}\n"
            "Run the download step first:\n"
            "  python -m molcrawl.preparation.preparation_script_protein_sequence"
            " assets/configs/protein_sequence.yaml --datasets proteingym --download-only"
        )

    # ------------------------------------------------------------------
    # 1. Collect sequences from all assay CSV files
    # ------------------------------------------------------------------
    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    logger.info("Found %d assay CSV files in %s", len(csv_files), source_dir)

    sequences: List[str] = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in {"mutated_sequence", "target_seq"})
        except Exception as exc:
            logger.warning("Skipping %s: %s", csv_path.name, exc)
            continue

        if "mutated_sequence" in df.columns:
            sequences.extend(df["mutated_sequence"].dropna().tolist())
        if "target_seq" in df.columns:
            sequences.extend(df["target_seq"].dropna().tolist())

    logger.info("Total sequences collected (before dedup): %d", len(sequences))

    # ------------------------------------------------------------------
    # 2. Deduplicate
    # ------------------------------------------------------------------
    sequences = list(dict.fromkeys(s for s in sequences if isinstance(s, str) and s.strip()))
    logger.info("Unique sequences after deduplication: %d", len(sequences))

    # ------------------------------------------------------------------
    # 3. Random 80 / 10 / 10 split
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sequences))

    n_train = int(len(idx) * train_ratio)
    n_val = int(len(idx) * (1 - train_ratio) / 2)

    train_seqs = [sequences[i] for i in idx[:n_train]]
    val_seqs = [sequences[i] for i in idx[n_train : n_train + n_val]]
    test_seqs = [sequences[i] for i in idx[n_train + n_val :]]

    logger.info(
        "Split — train: %d, valid: %d, test: %d",
        len(train_seqs),
        len(val_seqs),
        len(test_seqs),
    )

    raw_split = DatasetDict(
        {
            "train": Dataset.from_dict({"sequence": train_seqs}),
            "valid": Dataset.from_dict({"sequence": val_seqs}),
            "test": Dataset.from_dict({"sequence": test_seqs}),
        }
    )

    # ------------------------------------------------------------------
    # 4. Tokenise
    # ------------------------------------------------------------------
    logger.info("Initialising EsmSequenceTokenizer...")
    tokenizer = EsmSequenceTokenizer()
    eos_token_id = tokenizer.eos_token_id
    logger.info("vocab_size=%d  eos_token_id=%d", tokenizer.vocab_size, eos_token_id)

    logger.info("Tokenising sequences...")

    def _tokenize_batch(examples):
        results = []
        for seq in examples["sequence"]:
            encoded = tokenizer(str(seq), add_special_tokens=True)
            results.append(encoded["input_ids"])
        return {"input_ids": results}

    tokenized = raw_split.map(
        _tokenize_batch,
        batched=True,
        batch_size=1000,
        remove_columns=["sequence"],
        num_proc=num_proc,
        desc="Tokenising",
    )

    # ------------------------------------------------------------------
    # 5. Concatenate into a single stream, then chunk
    # ------------------------------------------------------------------
    logger.info("Concatenating and chunking to length %d...", context_length)

    concatenated = tokenized.map(
        partial(_concatenate_texts, eos_token_id=eos_token_id),
        batched=True,
        batch_size=-1,
        remove_columns=tokenized["train"].column_names,
        desc="Concatenating",
    )

    chunked = concatenated.map(
        partial(_create_chunks, context_length=context_length),
        batched=True,
        batch_size=-1,
        desc="Chunking",
    )

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    output_path = output_dir / "training_ready_hf_dataset"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving dataset to %s", output_path)
    chunked.save_to_disk(str(output_path))

    logger.info("Done! Dataset statistics:")
    for split_name in chunked:
        logger.info(
            "  %s: %d chunks of %d tokens",
            split_name,
            len(chunked[split_name]),
            context_length,
        )

    return str(output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import logging

    from molcrawl.core.paths import PROTEINGYM_DATASET_DIR, PROTEINGYM_SOURCE_DIR
    from molcrawl.protein_sequence.utils.configs import ProteinSequenceConfig

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = ArgumentParser(description="Prepare ProteinGym dataset for LM training")
    parser.add_argument("config", help="Path to protein_sequence YAML config file")
    args = parser.parse_args()

    # Use paths from paths.py constants; config is kept for future extensibility
    ProteinSequenceConfig.from_file(args.config)  # validate config exists

    prepare_proteingym(
        source_dir=PROTEINGYM_SOURCE_DIR,
        output_dir=str(PROTEINGYM_DATASET_DIR).replace("/training_ready_hf_dataset", ""),
    )
