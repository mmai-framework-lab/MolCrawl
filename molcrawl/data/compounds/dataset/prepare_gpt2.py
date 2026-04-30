import os
from argparse import ArgumentParser
from pathlib import Path

# Add project root src directory to path

# datasetLoad cache settings (assets/configs/cache.yamlfrom)
try:
    # Any cache settings. Learning can continue even in non-existent environments.
    from molcrawl.core.utils.cache_config import setup_cache_env
except ModuleNotFoundError:
    setup_cache_env = None

if setup_cache_env is not None:
    setup_cache_env()
else:
    # Can operate even in an environment without cache_config
    print("WARNING: utils.cache_config not found. Continuing without cache setup.")

from molcrawl.data.compounds.utils.config import CompoundConfig
from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer


def concatenate_texts(examples, eos_token_id):
    """Concatenate all input_ids sequences, appending eos_token_id after each."""
    concatenated_ids = []
    for input_ids in examples["input_ids"]:
        concatenated_ids.extend(list(input_ids) + [eos_token_id])
    return {"input_ids": concatenated_ids}


def create_chunks(examples, context_length):
    """Split a flat input_ids list into fixed-length chunks."""
    concatenated_ids = examples["input_ids"]
    total_length = (len(concatenated_ids) // context_length) * context_length
    concatenated_ids = concatenated_ids[:total_length]
    input_ids = [concatenated_ids[i : i + context_length] for i in range(0, total_length, context_length)]
    return {"input_ids": input_ids}


def tokenize_batch_dataset(compounds_dir, vocab_path, max_length):
    """
    Tokenize GuacaMol benchmark data for GPT-2 training.

    Each SMILES is encoded without padding; all sequences are concatenated with
    [SEP] (eos_token_id=13) as the end-of-sequence marker and chunked into
    blocks of 1024 tokens — matching the genome_sequence / protein_sequence
    preparation pipeline.

    Args:
        compounds_dir: Base directory for compounds data (from LEARNING_SOURCE_DIR)
        vocab_path: Path to vocabulary file
        max_length: Maximum token length per SMILES (used for truncation)
    """
    from functools import partial

    from datasets import Dataset, DatasetDict

    tokenizer = CompoundsTokenizer(vocab_path, max_length)

    # GuacaMol benchmark data directory
    benchmark_dir = Path(compounds_dir) / "benchmark" / "GuacaMol"

    dataset_dic = {}
    for split in ["train", "valid", "test"]:
        smiles_file = benchmark_dir / f"guacamol_v1_{split}.smiles"

        if not smiles_file.exists() or smiles_file.stat().st_size == 0:
            raise FileNotFoundError(
                f"GuacaMol benchmark file not found: {smiles_file}\n\n"
                f"Please download GuacaMol data by running:\n"
                f"  LEARNING_SOURCE_DIR={os.environ.get('LEARNING_SOURCE_DIR', 'learning_source_YYYYMMDD')} "
                f"python src/preparation/preparation_script_compounds.py assets/configs/compounds.yaml "
                f"--datasets guacamol --download-only\n\n"
                f"Or download manually from: https://figshare.com/projects/GuacaMol/56639\n"
                f"And place the files in: {benchmark_dir}/"
            )

        print(f"Loading {split} data from: {smiles_file}")
        with open(smiles_file) as f:
            lines = [line.strip() for line in f if line.strip()]

        # Encode without padding; [SEP] is appended per-sequence by concatenate_texts
        encoded = []
        for smi in lines:
            ids = tokenizer.encode(smi, add_special_tokens=False, truncation=True, max_length=max_length)
            if ids:
                encoded.append(ids)

        if not encoded:
            raise ValueError(
                f"No valid SMILES encoded from {smiles_file}. The file may be empty or all entries were filtered out."
            )
        print(f"{split} - {len(encoded)} molecules encoded; first decoded: {tokenizer.decode(encoded[0])}")
        dataset_dic[split] = encoded

    d = {
        "train": Dataset.from_dict({"input_ids": dataset_dic["train"]}),
        "valid": Dataset.from_dict({"input_ids": dataset_dic["valid"]}),
        "test": Dataset.from_dict({"input_ids": dataset_dic["test"]}),
    }
    dataset = DatasetDict(d)

    # Concatenate sequences with [SEP] (id=13) as EOS, then chunk into 1024-token blocks
    context_length = 1024
    eos_id = tokenizer.eos_token_id  # 13 ([SEP])

    concatenated = dataset.map(
        partial(concatenate_texts, eos_token_id=eos_id),
        batched=True,
        batch_size=-1,
    )
    chunked = concatenated.map(
        partial(create_chunks, context_length=context_length),
        batched=True,
        batch_size=-1,
    )

    # Save to compounds directory structure
    output_path = benchmark_dir / "compounds" / "training_ready_hf_dataset"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving dataset to: {output_path}")
    print("Match this path to the train_gpt2_config.py->dataset_dir parameter.")
    chunked.save_to_disk(str(output_path))

    # Print statistics
    print("\nDataset statistics:")
    for split in ["train", "valid", "test"]:
        print(f"  {split}: {len(chunked[split])} chunks of {context_length} tokens")


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
