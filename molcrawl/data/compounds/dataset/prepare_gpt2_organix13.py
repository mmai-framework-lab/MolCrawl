"""
GPT-2 training dataset preparation script (all compound dataset integrated version)

Supports v2 dataset configuration and from separate tokenized parquet files.
Convert to HuggingFace Dataset format for GPT-2 training.

Output path: {compounds_dir}/organix13/compounds/training_ready_hf_dataset
- GPT-2 training configuration (train_gpt2_config.py) references this path.

How to use:
  LEARNING_SOURCE_DIR=learning_source_YYYYMMDD python src/compounds/dataset/prepare_gpt2_organix13.py assets/configs/compounds.yaml
"""

import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from molcrawl.data.compounds.dataset.dataset_config import (
    DATASET_DEFINITIONS,
    CompoundDatasetType,
)
from molcrawl.data.compounds.utils.config import CompoundConfig


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


def prepare_gpt2_dataset(compounds_dir: str):
    """
    Integrate v2's individual tokenized parquet and create a GPT-2 training dataset.

    Processing details:
    1. Load parquet of all datasets from tokenized/ directory
    2. Extract and integrate the tokens column
    3. Rename tokens → input_ids
    4. Split into train/valid/test (80/10/10)
    5. Save in HuggingFace Dataset format

        Args:
    compounds_dir: compounds directorypath of
    """
    compounds_path = Path(compounds_dir)

    # Target all datasets excluding GuacaMol which is not for GPT-2
    target_datasets = [dt for dt in DATASET_DEFINITIONS.keys() if dt != CompoundDatasetType.GUACAMOL]

    # Collect tokenized parquet
    all_tokens = []
    loaded_datasets = []

    for dataset_type in target_datasets:
        info = DATASET_DEFINITIONS[dataset_type]
        tokenized_path = info.get_tokenized_path(compounds_path)

        if not tokenized_path.exists():
            print(f"  ⚠ {info.name}: Tokenized data not found at {tokenized_path}, skipping")
            continue

        print(f"  Loading {info.name} from {tokenized_path}...")
        table = pq.read_table(tokenized_path, columns=["tokens"])
        df = table.to_pandas()
        print(f"    → {len(df)} samples")

        all_tokens.append(df)
        loaded_datasets.append(info.name)

    if not all_tokens:
        raise FileNotFoundError(
            f"No tokenized datasets found in {compounds_path / 'tokenized'}\n\n"
            f"Please run the preparation script first:\n"
            f"  LEARNING_SOURCE_DIR={os.environ.get('LEARNING_SOURCE_DIR', 'learning_source_YYYYMMDD')} "
            f"python src/preparation/preparation_script_compounds.py assets/configs/compounds.yaml"
        )

    # Integration
    print(f"\nCombining {len(loaded_datasets)} datasets: {loaded_datasets}")
    combined_df = pd.concat(all_tokens, ignore_index=True)
    print(f"Total combined samples: {len(combined_df)}")

    # Rename tokens → input_ids (because GPT-2 learning expects input_ids)
    combined_df = combined_df.rename(columns={"tokens": "input_ids"})

    # Split into train/valid/test (80/10/10) at molecule level (before chunking)
    train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Split (molecules): Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")

    # Strip padding/[CLS]/[SEP] from each padded BERT sequence:
    #   [12(CLS), t1..tn, 13(SEP), 0(PAD)..] → [t1..tn]
    # Then concatenate with [SEP](id=13) as EOS and chunk into 1024-token blocks,
    # matching the genome_sequence / protein_sequence pretraining pipeline.
    def _strip_special(ids, cls_id=12, sep_id=13, pad_id=0):
        ids = list(ids)
        if ids and ids[0] == cls_id:
            ids = ids[1:]  # remove leading [CLS]
        while ids and ids[-1] == pad_id:
            ids.pop()  # remove trailing [PAD]
        if ids and ids[-1] == sep_id:
            ids.pop()  # remove [SEP] (re-added by _concat_with_eos)
        return ids

    from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer

    tokenizer = CompoundsTokenizer("assets/molecules/vocab.txt", 256)
    eos_id = tokenizer.eos_token_id  # 13 ([SEP])
    context_length = 1024

    def _make_chunked_split(df):
        raw = [_strip_special(row) for row in df["input_ids"]]
        raw = [ids for ids in raw if ids]  # drop any empty sequences
        ds = Dataset.from_dict({"input_ids": raw})
        ds = ds.map(partial(_concat_with_eos, eos_token_id=eos_id), batched=True, batch_size=-1)
        ds = ds.map(partial(_create_chunks, context_length=context_length), batched=True, batch_size=-1)
        return ds

    print("\nConcatenating and chunking into 1024-token blocks …")
    dataset = DatasetDict(
        {
            "train": _make_chunked_split(train_df),
            "valid": _make_chunked_split(valid_df),
            "test": _make_chunked_split(test_df),
        }
    )

    # Save to the legacy output path for backward compatibility with GPT-2 training configs
    # (train_gpt2_config.py references: compounds/organix13/compounds/training_ready_hf_dataset)
    output_path = compounds_path / "organix13" / "compounds" / "training_ready_hf_dataset"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving dataset to: {output_path}")
    dataset.save_to_disk(str(output_path))

    # Print statistics
    print("\nDataset statistics (1024-token chunks):")
    for split in ["train", "valid", "test"]:
        print(f"  {split}: {len(dataset[split])} chunks")
    print("\nThis path matches train_gpt2_config.py → dataset_dir parameter.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare GPT-2 training dataset from tokenized compounds data")
    parser.add_argument(
        "config",
        help="Path to compounds config file (e.g. assets/configs/compounds.yaml)",
    )
    args = parser.parse_args()

    # config is received for compatibility, but v2 uses tokenized data directly
    # vocab_path / max_length is not required
    _ = CompoundConfig.from_file(args.config)

    # Get compounds directory from LEARNING_SOURCE_DIR
    learning_source_dir = os.environ.get("LEARNING_SOURCE_DIR")
    if not learning_source_dir:
        raise ValueError(
            "LEARNING_SOURCE_DIR environment variable is not set.\n"
            "Please set it before running this script:\n"
            "  export LEARNING_SOURCE_DIR='learning_source_YYYYMMDD'"
        )

    compounds_dir = Path(learning_source_dir) / "compounds"
    print(f"Using compounds directory: {compounds_dir}")

    prepare_gpt2_dataset(str(compounds_dir))
