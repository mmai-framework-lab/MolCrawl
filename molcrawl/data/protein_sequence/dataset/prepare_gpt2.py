from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

# add project root to path（utilsetc.)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# datasetLoad cache settings (assets/configs/cache.yamlfrom)
try:
    from molcrawl.core.utils.cache_config import setup_cache_env
except ModuleNotFoundError:
    setup_cache_env = None

if setup_cache_env is not None:
    setup_cache_env()
else:
    # Can operate even in an environment without cache_config
    print("WARNING: utils.cache_config not found. Continuing without cache setup.")

from molcrawl.data.protein_sequence.utils.configs import ProteinSequenceConfig  # noqa: E402

if TYPE_CHECKING:
    from molcrawl.data.protein_sequence.dataset.tokenizer import EsmSequenceTokenizer


def tokenize_function(examples: Dict[str, List[str]], tokenizer: EsmSequenceTokenizer) -> Dict[str, List[List[int]]]:
    return {
        "input_ids": tokenizer(
            examples["text"],
            truncation=False,
            add_special_tokens=False,  # We'll add special tokens manually
        )["input_ids"]
    }


def concatenate_texts(examples: Dict[str, List[List[int]]], eos_token_id: int) -> Dict[str, List[int]]:
    concatenated_ids: List[int] = []
    for input_ids in examples["input_ids"]:
        concatenated_ids.extend(input_ids + [eos_token_id])
    return {"input_ids": concatenated_ids}


def create_chunks(examples: Dict[str, List[int]], context_length: int) -> Dict[str, List[List[int]]]:
    concatenated_ids: List[int] = examples["input_ids"]

    # Calculate the total number of chunks
    total_length = len(concatenated_ids)
    num_chunks = total_length // context_length

    # Truncate the concatenated_ids to a multiple of context_length
    total_length = num_chunks * context_length
    concatenated_ids = concatenated_ids[:total_length]

    # Split into chunks
    input_ids = [concatenated_ids[i : i + context_length] for i in range(0, total_length, context_length)]

    return {"input_ids": input_ids}


def tokenize_batch_dataset(path_output: Path, context_length: int, number_sample: int) -> None:
    from datasets import DatasetDict, load_dataset

    from molcrawl.data.protein_sequence.dataset.tokenizer import EsmSequenceTokenizer

    raw_dir: Path = Path(path_output) / "raw_files"
    raw_files: List[Path] = sorted(raw_dir.glob("*.raw")) + sorted(raw_dir.glob("*.txt"))
    if not raw_files:
        raise FileNotFoundError(
            f"No raw data files found in {raw_dir}. "
            "Expected *.raw or *.txt files. Check symlinks or rerun the preparation step."
        )

    # Avoid the effect of extension determination by explicitly passing the file list
    data = (
        load_dataset(
            "text",
            data_files={"train": [str(p) for p in raw_files]},
            split="train",
        )
        .shuffle()
        .select(range(number_sample))
    )
    raw_datasets = data.train_test_split(test_size=0.2)
    valid_test_split = raw_datasets["test"].train_test_split(test_size=0.5)
    raw_datasets = DatasetDict(
        {"train": raw_datasets["train"], "valid": valid_test_split["train"], "test": valid_test_split["test"]}
    )

    tokenizer: EsmSequenceTokenizer = EsmSequenceTokenizer()

    tokenized_datasets = raw_datasets.map(
        partial(tokenize_function, tokenizer=tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    concatenated_dataset = tokenized_datasets.map(
        partial(concatenate_texts, eos_token_id=tokenizer.eos_token_id),
        batched=True,
        batch_size=-1,
    )

    chunked_dataset = concatenated_dataset.map(
        partial(create_chunks, context_length=context_length),
        batched=True,
        batch_size=-1,
    )

    path_dataset: str = str(path_output / "training_ready_hf_dataset")
    print(f"Saving dataset to: {path_dataset}. Match this path to the train_gpt2_config.py->dataset_dir parameter.")
    chunked_dataset.save_to_disk(path_dataset)


if __name__ == "__main__":
    number_sample: int = 50000
    context_length: int = 1024

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = ProteinSequenceConfig.from_file(args.config).data_preparation

    output_dir: Path = Path(cfg.output_dir)
    tokenize_batch_dataset(output_dir, context_length, number_sample)
