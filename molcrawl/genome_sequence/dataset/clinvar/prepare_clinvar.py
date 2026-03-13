"""
ClinVar Dataset Preparation Script for GPT-2 / BERT Fine-tuning

Converts the project-level ``dataset/clinvar_sequences.csv`` into the
training_ready_hf_dataset format compatible with the genome_sequence
training pipeline.

Pipeline:
    1. Read ``reference_sequence`` and ``variant_sequence`` columns from
       the ClinVar CSV (Benign / Pathogenic variants on human chromosomes).
    2. Deduplicate by exact sequence string.
    3. Shuffle and apply an 80 / 10 / 10 train / valid / test split.
    4. Tokenise with the genome SentencePiece BPE tokenizer (vocab_size=4096).
    5. Concatenate all token sequences into one long stream (with EOS between
       sequences), then chunk into fixed-length blocks of *context_length*.
    6. Save as a HuggingFace DatasetDict to
       ``CLINVAR_DIR/training_ready_hf_dataset/``.

Usage (standalone):
    LEARNING_SOURCE_DIR=learning_source_20260311 \\
    python -m molcrawl.genome_sequence.dataset.clinvar.prepare_clinvar

    The output is saved to:
        $LEARNING_SOURCE_DIR/genome_sequence/clinvar/training_ready_hf_dataset/
"""

import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-level helpers (top-level functions for num_proc > 1 compatibility)
# ---------------------------------------------------------------------------


def _tokenize_batch(examples: Dict, tokenizer_path: str) -> Dict:
    """Tokenise a batch of DNA sequences using SentencePiece BPE."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    results = []
    for seq in examples["sequence"]:
        results.append(sp.encode(str(seq).upper()))
    return {"input_ids": results}


def _concatenate_texts(examples: Dict, eos_token_id: int) -> Dict:
    """Concatenate all input_ids in a batch into one flat list with EOS separators."""
    all_ids: List[int] = []
    for ids in examples["input_ids"]:
        all_ids.extend(ids)
        all_ids.append(eos_token_id)
    return {"input_ids": all_ids}


def _create_chunks(examples: Dict, context_length: int) -> Dict:
    """Split a flat input_ids list into fixed-length blocks."""
    ids = examples["input_ids"]
    n_chunks = len(ids) // context_length
    chunks = [ids[i * context_length : (i + 1) * context_length] for i in range(n_chunks)]
    return {"input_ids": chunks}


# ---------------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------------


def prepare_clinvar(
    source_file: Union[str, Path],
    output_dir: Union[str, Path],
    tokenizer_path: str,
    *,
    context_length: int = 1024,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_proc: int = 4,
) -> str:
    """
    Load ClinVar sequences from *source_file*, build a language-model
    training dataset, and save it to *output_dir*.

    Args:
        source_file: Path to ``clinvar_sequences.csv``.
        output_dir: Parent directory; the dataset is written under
                    ``output_dir/training_ready_hf_dataset/``.
        tokenizer_path: Path to the SentencePiece ``.model`` file.
        context_length: Token block length (default 1024).
        train_ratio: Fraction of sequences used for training (rest split 50/50
                     between validation and test).
        seed: Random seed for reproducible shuffling.
        num_proc: Parallel workers for HuggingFace Dataset.map() operations.

    Returns:
        Absolute path string of the saved dataset directory.
    """
    import csv

    import sentencepiece as spm

    source_file = Path(source_file)
    output_dir = Path(output_dir)

    if not source_file.exists():
        raise FileNotFoundError(
            f"ClinVar source file not found: {source_file}\n" "Expected at: dataset/clinvar_sequences.csv in the project root."
        )

    # ------------------------------------------------------------------
    # 1. Collect sequences from CSV
    # ------------------------------------------------------------------
    logger.info("Reading ClinVar sequences from %s", source_file)
    sequences: List[str] = []
    with open(source_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_seq = row.get("reference_sequence", "").strip()
            var_seq = row.get("variant_sequence", "").strip()
            if ref_seq:
                sequences.append(ref_seq.upper())
            if var_seq:
                sequences.append(var_seq.upper())

    logger.info("Total sequences collected (before dedup): %d", len(sequences))

    # ------------------------------------------------------------------
    # 2. Deduplicate
    # ------------------------------------------------------------------
    sequences = list(dict.fromkeys(s for s in sequences if s))
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
    # 4. Tokenise with SentencePiece BPE genome tokenizer
    # ------------------------------------------------------------------
    logger.info("Tokenising sequences with SentencePiece BPE from %s", tokenizer_path)
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    eos_token_id = sp.eos_id()
    logger.info("vocab_size=%d  eos_token_id=%d", sp.get_piece_size(), eos_token_id)

    tokenized = raw_split.map(
        partial(_tokenize_batch, tokenizer_path=tokenizer_path),
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

    from molcrawl.config.paths import CLINVAR_DIR, CLINVAR_SOURCE_FILE, get_refseq_tokenizer_path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    prepare_clinvar(
        source_file=CLINVAR_SOURCE_FILE,
        output_dir=CLINVAR_DIR,
        tokenizer_path=get_refseq_tokenizer_path(),
    )
