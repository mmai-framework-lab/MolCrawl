from argparse import ArgumentParser
import os
import sys
from pathlib import Path

# プロジェクトルートのsrcディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from compounds.utils.config import CompoundConfig
from compounds.utils.tokenizer import CompoundsTokenizer
from datasets import Dataset, DatasetDict


def tokenize_batch_dataset(compounds_dir, vocab_path, max_length):
    """
    Tokenize OrganiX13 parquet data for GPT-2 training.

    Args:
        compounds_dir: Base directory for compounds data (from LEARNING_SOURCE_DIR)
        vocab_path: Path to vocabulary file
        max_length: Maximum token length
    """
    tokenizer = CompoundsTokenizer(
        vocab_path,
        max_length,
    )

    # OrganiX13 parquet file location
    organix13_dir = Path(compounds_dir) / "organix13"
    organix13_file = organix13_dir / "OrganiX13.parquet"

    # Check if OrganiX13.parquet exists
    if not organix13_file.exists():
        raise FileNotFoundError(
            f"OrganiX13 parquet file not found: {organix13_file}\n\n"
            f"Please run the preparation script first:\n"
            f"  LEARNING_SOURCE_DIR={os.environ.get('LEARNING_SOURCE_DIR', 'learning_20251209')} "
            f"python scripts/preparation/preparation_script_compounds.py assets/configs/compounds.yaml"
        )

    print(f"Loading OrganiX13 data from: {organix13_file}")
    df = pd.read_parquet(organix13_file)

    # Check if 'smiles' column exists
    if "smiles" not in df.columns:
        raise ValueError(f"'smiles' column not found in {organix13_file}")

    print(f"Total samples: {len(df)}")

    # Split into train/valid/test (80/10/10)
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    # Tokenize each split
    dataset_dic = {}
    for split, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        print(f"Tokenizing {len(split_df)} SMILES for {split} split...")
        split_df["tokens"] = split_df["smiles"].apply(tokenizer.tokenize_text)
        print(f"Example decoded: {tokenizer.decode(split_df['tokens'].iloc[0])}")
        dataset_dic[split] = split_df

    # Create HuggingFace Dataset
    d = {
        "train": Dataset.from_dict({"input_ids": dataset_dic["train"]["tokens"].to_numpy()}),
        "valid": Dataset.from_dict({"input_ids": dataset_dic["valid"]["tokens"].to_numpy()}),
        "test": Dataset.from_dict({"input_ids": dataset_dic["test"]["tokens"].to_numpy()}),
    }

    dataset = DatasetDict(d)

    # Save to compounds directory structure (OrganiX13 specific path)
    output_path = organix13_dir / "compounds" / "training_ready_hf_dataset"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving dataset to: {output_path}")
    print("Match this path to the train_gpt2_config.py->dataset_dir parameter.")
    dataset.save_to_disk(str(output_path))

    # Print statistics
    print("\nDataset statistics:")
    for split in ["train", "valid", "test"]:
        print(f"  {split}: {len(dataset[split])} samples")


if __name__ == "__main__":
    number_sample = None

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = CompoundConfig.from_file(args.config).data_preparation
    context_length = cfg.max_length

    # Get compounds directory from LEARNING_SOURCE_DIR
    learning_source_dir = os.environ.get("LEARNING_SOURCE_DIR")
    if not learning_source_dir:
        raise ValueError(
            "LEARNING_SOURCE_DIR environment variable is not set.\n"
            "Please set it before running this script:\n"
            "  export LEARNING_SOURCE_DIR='learning_20251210'"
        )

    compounds_dir = Path(learning_source_dir) / "compounds"
    print(f"Using compounds directory: {compounds_dir}")

    tokenize_batch_dataset(str(compounds_dir), cfg.vocab_path, cfg.max_length)
