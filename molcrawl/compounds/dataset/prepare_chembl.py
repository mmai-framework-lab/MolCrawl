#!/usr/bin/env python3
"""
Prepare ChEMBL for GPT-2 / BERT fine-tuning on the compounds domain.

Reads the canonical SMILES file produced by ``download_chembl.py``, tokenises
each SMILES string with the same ``CompoundsTokenizer`` used during pretraining,
shuffles the dataset, splits it 80 / 10 / 10 (train / valid / test), and saves
the result in HuggingFace Dataset format to *output_dir*.

The output directory layout is compatible with ``molcrawl/core/dataset.py``
(``PreparedDataset``) and with the HuggingFace ``Trainer`` used in
``molcrawl/bert/main.py``:

    training_ready_hf_dataset/
        dataset_info.json
        train/
        valid/
        test/

Usage
─────
    python -m molcrawl.compounds.dataset.prepare_chembl       # uses CHEMBL_* constants
    python -m molcrawl.compounds.dataset.prepare_chembl --force
"""

import logging
import sys
from functools import partial
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

VOCAB_FILE = "assets/molecules/vocab.txt"
MAX_LEN = 256
BATCH_SIZE = 50_000  # rows per batch when tokenising


def _concat_with_eos(examples, eos_token_id):
    """Concatenate all input_ids sequences, appending eos_token_id after each."""
    concatenated_ids = []
    for input_ids in examples["input_ids"]:
        concatenated_ids.extend(list(input_ids) + [eos_token_id])
    return {"input_ids": [concatenated_ids]}


def _create_chunks(examples, context_length):
    """Split a flat input_ids list into fixed-length chunks."""
    concatenated_ids = examples["input_ids"]
    total_length = len(concatenated_ids)
    num_chunks = total_length // context_length
    total_length = num_chunks * context_length
    concatenated_ids = concatenated_ids[:total_length]
    input_ids = [concatenated_ids[i : i + context_length] for i in range(0, total_length, context_length)]
    return {"input_ids": input_ids}


def _tokenize_batch(smiles_list: List[str], tokenizer) -> List[Optional[List[int]]]:
    """Tokenise SMILES without padding; returns raw token ids (no [CLS]/[SEP]).
    ``None`` entries indicate invalid/empty SMILES."""
    results = []
    for smi in smiles_list:
        try:
            # encode without special tokens or padding; concatenation pipeline adds [SEP]
            ids = tokenizer.encode(
                smi.strip(),
                add_special_tokens=False,
                truncation=True,
                max_length=tokenizer.max_len,
            )
            results.append(ids if ids else None)
        except Exception:
            results.append(None)
    return results


def prepare_chembl(
    source_dir: str,
    output_dir: str,
    vocab_file: str = VOCAB_FILE,
    max_len: int = MAX_LEN,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    random_seed: int = 42,
    force: bool = False,
    num_proc: int = 4,
) -> bool:
    """Tokenise ChEMBL SMILES and save a HuggingFace DatasetDict.

    Args:
        source_dir: Directory containing ``smiles.txt`` (output of
            ``download_chembl.py``).
        output_dir: Destination for ``training_ready_hf_dataset/``.
        vocab_file: Path to the SMILES WordPiece vocabulary file.
        max_len: Maximum token length (same as pretraining, default 256).
        train_ratio: Fraction of data used for training.
        valid_ratio: Fraction for validation (remainder → test).
        random_seed: Random seed for shuffling.
        force: Re-run even if the marker file already exists.
        num_proc: Worker count for HuggingFace ``.map()`` parallelism.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    from datasets import Dataset, DatasetDict

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    marker = out / "prepare_complete.marker"
    if not force and marker.exists():
        logger.info("ChEMBL preparation already completed. Skipping. (use force=True to re-run)")
        return True

    smiles_file = Path(source_dir) / "smiles.txt"
    if not smiles_file.exists():
        logger.error(f"smiles.txt not found at {smiles_file}. Run download_chembl.py first.")
        return False

    # ── 1. Read SMILES ────────────────────────────────────────────────────────
    logger.info(f"Reading SMILES from {smiles_file} …")
    with smiles_file.open("r") as fh:
        smiles_list = [line.strip() for line in fh if line.strip()]
    logger.info(f"Loaded {len(smiles_list):,} SMILES strings.")

    # ── 2. Tokenise in batches ────────────────────────────────────────────────
    from molcrawl.compounds.utils.tokenizer import CompoundsTokenizer

    tokenizer = CompoundsTokenizer(vocab_file, max_len)

    logger.info(f"Tokenising with vocab_size={tokenizer.vocab_size}, max_len={max_len} …")
    all_input_ids: List[List[int]] = []
    skipped = 0

    for start in range(0, len(smiles_list), BATCH_SIZE):
        batch = smiles_list[start : start + BATCH_SIZE]
        tokenised = _tokenize_batch(batch, tokenizer)
        for ids in tokenised:
            if ids is None:
                skipped += 1
            else:
                all_input_ids.append(ids)
        if (start // BATCH_SIZE) % 10 == 0:
            logger.info(f"  … {start + len(batch):,} / {len(smiles_list):,} processed ({skipped:,} skipped so far)")

    logger.info(
        f"Tokenisation complete. "
        f"Kept {len(all_input_ids):,}, skipped {skipped:,} "
        f"({skipped * 100 / max(1, len(smiles_list)):.1f}%)."
    )

    if not all_input_ids:
        logger.error("No valid SMILES tokenised. Aborting.")
        return False

    # ── 3. Build Dataset and split ────────────────────────────────────────────
    logger.info("Building HuggingFace Dataset …")
    dataset = Dataset.from_dict({"input_ids": all_input_ids})
    dataset = dataset.shuffle(seed=random_seed)

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_ds = dataset.select(range(n_train))
    valid_ds = dataset.select(range(n_train, n_train + n_valid))
    test_ds = dataset.select(range(n_train + n_valid, n))

    logger.info(f"Split: train={len(train_ds):,}, valid={len(valid_ds):,}, test={len(test_ds):,}")

    # ── 3b. Concatenate with [SEP] as EOS and chunk into 1024-token blocks ────
    # Matches the genome_sequence / protein_sequence pretraining pipeline.
    from molcrawl.compounds.utils.tokenizer import CompoundsTokenizer as _Tok

    _tok = _Tok(vocab_file, max_len)
    eos_id = _tok.eos_token_id  # 13 ([SEP])
    context_length = 1024

    logger.info(f"Concatenating and chunking into {context_length}-token blocks (eos_id={eos_id}) …")
    chunked_splits = {}
    for split_name, split_ds in [("train", train_ds), ("valid", valid_ds), ("test", test_ds)]:
        split_ds = split_ds.map(partial(_concat_with_eos, eos_token_id=eos_id), batched=True, batch_size=-1)
        split_ds = split_ds.map(partial(_create_chunks, context_length=context_length), batched=True, batch_size=-1)
        chunked_splits[split_name] = split_ds
        logger.info(f"  {split_name}: {len(split_ds):,} chunks")
    dataset_dict = DatasetDict(chunked_splits)

    # ── 4. Save ──────────────────────────────────────────────────────────────
    hf_path = out / "training_ready_hf_dataset"
    logger.info(f"Saving dataset to {hf_path} …")
    dataset_dict.save_to_disk(str(hf_path))
    for split_name, split_ds in dataset_dict.items():
        logger.info(f"  Saved {split_name}: {len(split_ds):,} chunks → {hf_path / split_name}")

    marker.touch()
    logger.info(f"ChEMBL preparation complete. Marker written to {marker}")
    return True


if __name__ == "__main__":
    from molcrawl.core.paths import CHEMBL_DIR, CHEMBL_SOURCE_DIR
    from molcrawl.core.base import setup_logging

    setup_logging(CHEMBL_DIR)
    force_flag = "--force" in sys.argv
    success = prepare_chembl(
        source_dir=CHEMBL_SOURCE_DIR,
        output_dir=CHEMBL_DIR,
        force=force_flag,
    )
    sys.exit(0 if success else 1)
