"""
Cell Type Annotation Dataset Preparation Script for GPT-2 / BERT Fine-tuning

Downloads the Geneformer cell type annotation example dataset from the
Genecorpus-30M repository (ctheodoris/Genecorpus-30M on HuggingFace) and
prepares it for language-model fine-tuning using the RNA tokenizer.

The dataset contains human single-cell transcriptomes already pre-tokenized
as rank-value encodings (same format as the CellxGene pretraining corpus).
No re-tokenization is required — only splitting and chunking.

Pipeline:
    1. Download cell_type_train_data.dataset from ctheodoris/Genecorpus-30M
       (example_input_files/cell_classification/cell_type_annotation/) via
       the HuggingFace Hub.
    2. Extract ``input_ids`` (gene token IDs, already tokenized by
       TranscriptomeTokenizer).
    3. Shuffle and apply an 80 / 10 / 10 train / valid / test split.
    4. Concatenate all token sequences into one long stream (with EOS between
       sequences), then chunk into fixed-length blocks of *context_length*.
    5. Save as a HuggingFace DatasetDict to
       ``RNA_CELLTYPE_DIR/training_ready_hf_dataset/``.

Usage (standalone):
    LEARNING_SOURCE_DIR=learning_source_20260311 \\
    python -m molcrawl.rna.dataset.celltype.prepare_celltype

    The output is saved to:
        $LEARNING_SOURCE_DIR/rna/celltype/training_ready_hf_dataset/
"""

import logging
import shutil
from functools import partial
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

# HuggingFace source
_HF_REPO_ID = "ctheodoris/Genecorpus-30M"
_HF_DATASET_SUBPATH = "example_input_files/cell_classification" "/cell_type_annotation/cell_type_train_data.dataset"
_DOWNLOADED_DIRNAME = "cell_type_train_data.dataset"


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def download_celltype(save_path: Union[str, Path]) -> Path:
    """
    Download the Geneformer cell type annotation dataset from HuggingFace Hub.

    The Arrow dataset directory is placed at ``save_path/cell_type_train_data
    .dataset/``.  If it already exists and is non-empty the download is skipped.

    Args:
        save_path: Local directory into which the dataset is downloaded.

    Returns:
        Path to the Arrow dataset directory.
    """
    from huggingface_hub import snapshot_download

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    dataset_dir = save_path / _DOWNLOADED_DIRNAME
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        logger.info("Cell type dataset already downloaded: %s", dataset_dir)
        return dataset_dir

    logger.info(
        "Downloading cell type annotation dataset from %s (this may take a while) ...",
        _HF_REPO_ID,
    )

    cache_dir = save_path / "_hf_cache"
    snapshot_download(
        repo_id=_HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"{_HF_DATASET_SUBPATH}/**"],
        local_dir=str(cache_dir),
    )

    # Move the nested downloaded directory to the expected flat location.
    downloaded_nested = cache_dir / Path(_HF_DATASET_SUBPATH)
    if downloaded_nested.exists():
        shutil.copytree(str(downloaded_nested), str(dataset_dir))
        shutil.rmtree(str(cache_dir))
        logger.info("Downloaded cell type dataset to %s", dataset_dir)
    else:
        raise FileNotFoundError(
            f"Expected dataset not found after download: {downloaded_nested}\n"
            "The repository layout may have changed; check "
            f"https://huggingface.co/datasets/{_HF_REPO_ID}"
        )

    return dataset_dir


# ---------------------------------------------------------------------------
# Token-level helpers (top-level for num_proc > 1 compatibility)
# ---------------------------------------------------------------------------


def _concatenate_texts(examples: Dict, eos_token_id: int) -> Dict:
    """Concatenate all input_ids in a batch into one flat list + EOS."""
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


def prepare_celltype(
    source_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    context_length: int = 1024,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_proc: int = 4,
    eos_token_id: int = 0,
) -> str:
    """
    Load the pre-tokenized Geneformer cell type dataset from *source_dir*,
    build a language-model training dataset, and save it to *output_dir*.

    The dataset is already tokenized (gene rank-value encodings produced by
    TranscriptomeTokenizer), so only splitting, concatenation, and chunking
    are performed.

    Args:
        source_dir: Directory containing the downloaded
                    ``cell_type_train_data.dataset/`` Arrow directory.
        output_dir: Parent directory; the dataset is written under
                    ``output_dir/training_ready_hf_dataset/``.
        context_length: Token block length (default 1024).
        train_ratio: Fraction used for training (rest split 50/50 between
                     validation and test).
        seed: Random seed for reproducible shuffling.
        num_proc: Parallel workers for HuggingFace Dataset.map() operations.
        eos_token_id: Token ID appended between sequences (default 0,
                      consistent with RNA BERT config ``eos_token = 0``).

    Returns:
        Absolute path string of the saved ``training_ready_hf_dataset/``
        directory.
    """
    from datasets import load_from_disk

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    dataset_arrow = source_dir / _DOWNLOADED_DIRNAME
    if not dataset_arrow.exists():
        raise FileNotFoundError(
            f"Cell type dataset not found: {dataset_arrow}\n"
            "Run the download step first:\n"
            "  python -m molcrawl.preparation.preparation_script_rna "
            "assets/configs/rna.yaml --datasets celltype"
        )

    # ------------------------------------------------------------------
    # 1. Load pre-tokenized HuggingFace Arrow dataset
    # ------------------------------------------------------------------
    logger.info("Loading cell type dataset from %s", dataset_arrow)
    raw_dataset = load_from_disk(str(dataset_arrow))

    # Unwrap DatasetDict if needed
    if hasattr(raw_dataset, "keys"):
        split_name = next(iter(raw_dataset.keys()))
        raw_dataset = raw_dataset[split_name]

    total = len(raw_dataset)
    logger.info("Loaded %d single-cell transcriptomes", total)

    # ------------------------------------------------------------------
    # 2. Extract input_ids (gene token IDs; labels discarded for LM
    #    fine-tuning)
    # ------------------------------------------------------------------
    all_ids: List[List[int]] = list(raw_dataset["input_ids"])

    # ------------------------------------------------------------------
    # 3. Random 80 / 10 / 10 split
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    idx = rng.permutation(total)

    n_train = int(total * train_ratio)
    n_val = int(total * (1 - train_ratio) / 2)

    train_ids = [all_ids[i] for i in idx[:n_train]]
    val_ids = [all_ids[i] for i in idx[n_train : n_train + n_val]]
    test_ids = [all_ids[i] for i in idx[n_train + n_val :]]

    logger.info(
        "Split — train: %d, valid: %d, test: %d",
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )

    raw_split = DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": train_ids}),
            "valid": Dataset.from_dict({"input_ids": val_ids}),
            "test": Dataset.from_dict({"input_ids": test_ids}),
        }
    )

    # ------------------------------------------------------------------
    # 4. Concatenate all sequences + chunk into fixed-length blocks
    # ------------------------------------------------------------------
    _concat_fn = partial(_concatenate_texts, eos_token_id=eos_token_id)
    _chunk_fn = partial(_create_chunks, context_length=context_length)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output = output_dir / "training_ready_hf_dataset"

    split_datasets = {}
    for sname, split_ds in raw_split.items():
        logger.info("Processing split '%s' (%d cells)...", sname, len(split_ds))

        flat = split_ds.map(
            _concat_fn,
            batched=True,
            batch_size=len(split_ds),
            remove_columns=split_ds.column_names,
            num_proc=1,  # full-batch concat requires a single worker
        )
        chunked = flat.map(
            _chunk_fn,
            batched=True,
            batch_size=1,
            remove_columns=flat.column_names,
            num_proc=1,
        )
        logger.info("  → %d chunks of length %d", len(chunked), context_length)
        split_datasets[sname] = chunked

    final_dataset = DatasetDict(split_datasets)

    logger.info("Saving chunked dataset to %s", dataset_output)
    final_dataset.save_to_disk(str(dataset_output))

    logger.info(
        "Done. Saved %d / %d / %d chunks (train / valid / test).",
        len(split_datasets["train"]),
        len(split_datasets["valid"]),
        len(split_datasets["test"]),
    )
    return str(dataset_output)


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    from molcrawl.config.paths import RNA_CELLTYPE_DIR, RNA_CELLTYPE_SOURCE_DIR

    logger.info("=== Cell Type Dataset Preparation ===")
    src = download_celltype(RNA_CELLTYPE_SOURCE_DIR)
    prepare_celltype(source_dir=RNA_CELLTYPE_SOURCE_DIR, output_dir=RNA_CELLTYPE_DIR)
    logger.info("Finished.")
