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
    Tokenize GuacaMol benchmark data for GPT-2 training.
    
    Args:
        compounds_dir: Base directory for compounds data (from LEARNING_SOURCE_DIR)
        vocab_path: Path to vocabulary file
        max_length: Maximum token length
    """
    tokenizer = CompoundsTokenizer(
        vocab_path,
        max_length,
    )

    # GuacaMol benchmark data directory
    benchmark_dir = Path(compounds_dir) / "benchmark" / "GuacaMol"
    
    dataset_dic = {}
    for split in ["train", "valid", "test"]:
        # Use relative path from compounds directory
        smiles_file = benchmark_dir / f"guacamol_v1_{split}.smiles"
        
        if not smiles_file.exists():
            raise FileNotFoundError(
                f"GuacaMol benchmark file not found: {smiles_file}\n\n"
                f"Please download GuacaMol data by running:\n"
                f"  LEARNING_SOURCE_DIR={os.environ.get('LEARNING_SOURCE_DIR', 'learning_20251104')} python scripts/preparation/download_guacamol.py\n\n"
                f"Or download manually from: https://figshare.com/projects/GuacaMol/56639\n"
                f"And place the files in: {benchmark_dir}/"
            )
        
        print(f"Loading {split} data from: {smiles_file}")
        with open(smiles_file) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line]
        
        df = pd.DataFrame(lines, columns=["smiles"])
        df["tokens"] = df["smiles"].apply(tokenizer.tokenize_text)
        print(f"{split} - First molecule decoded: {tokenizer.decode(df['tokens'].iloc[0])}")
        dataset_dic[split] = df

    d = {
        "train": Dataset.from_dict(
            {"input_ids": dataset_dic["train"]["tokens"].to_numpy()}
        ),
        "valid": Dataset.from_dict(
            {"input_ids": dataset_dic["valid"]["tokens"].to_numpy()}
        ),
        "test": Dataset.from_dict(
            {"input_ids": dataset_dic["test"]["tokens"].to_numpy()}
        ),
    }

    dataset = DatasetDict(d)

    # Save to compounds directory structure
    output_path = benchmark_dir / "compounds" / "training_ready_hf_dataset"
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
            "  export LEARNING_SOURCE_DIR='learning_20251104'"
        )
    
    compounds_dir = Path(learning_source_dir) / "compounds"
    print(f"Using compounds directory: {compounds_dir}")

    tokenize_batch_dataset(str(compounds_dir), cfg.vocab_path, cfg.max_length)
