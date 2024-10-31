from functools import partial
from argparse import ArgumentParser
from pathlib import Path

from datasets import DatasetDict, Dataset
import numpy as np

from compounds.utils.tokenizer import CompoundsTokenizer
from compounds.utils.config import CompoundConfig
from core.base import read_parquet


def concatenate_texts(examples, eos_token_id):
    concatenated_ids = []
    for input_ids in examples["input_ids"]:
        if input_ids is None:
            continue
        concatenated_ids.extend(input_ids + [eos_token_id])
    return {"input_ids": concatenated_ids}


def create_chunks(examples, context_length):
    concatenated_ids = examples["input_ids"]

    # Calculate the total number of chunks
    total_length = len(concatenated_ids)
    num_chunks = total_length // context_length

    # Truncate the concatenated_ids to a multiple of context_length
    total_length = num_chunks * context_length
    concatenated_ids = concatenated_ids[:total_length]

    # Split into chunks
    input_ids = [concatenated_ids[i : i + context_length] for i in range(0, total_length, context_length)]

    return {"input_ids": input_ids}


def tokenize_batch_dataset(parquet_path, vocab_path, max_length, context_length, number_sample):

    tokenize_dataset = read_parquet(parquet_path)

    indices = np.arange(len(tokenize_dataset))
    np.random.shuffle(indices)

    d = {
        "train": Dataset.from_dict({"input_ids": tokenize_dataset["tokens"].to_numpy()[indices[: int(number_sample * 0.8)]]}),
        "valid": Dataset.from_dict({"input_ids": tokenize_dataset["tokens"].to_numpy()[indices[int(number_sample * 0.8) : int(number_sample * 0.9)]]}),
        "test": Dataset.from_dict({"input_ids": tokenize_dataset["tokens"].to_numpy()[indices[int(number_sample * 0.9) :]]}),
    }

    dataset = DatasetDict(d)

    tokenizer = CompoundsTokenizer(
        vocab_path,
        max_length,
    )

    concatenated_dataset = dataset.map(
        partial(concatenate_texts, eos_token_id=tokenizer.eos_token_id),
        batched=True,
        batch_size=-1,
    )

    chunked_dataset = concatenated_dataset.map(
        partial(create_chunks, context_length=context_length),
        batched=True,
        batch_size=-1,
    )

    path_dataset = str(Path(parquet_path).parent / "compounds" / "training_ready_hf_dataset")
    chunked_dataset.save_to_disk(path_dataset)


if __name__ == "__main__":
    number_sample = 50000
    context_length = 1024

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = CompoundConfig.from_file(args.config).data_preparation

    tokenize_batch_dataset(cfg.save_path, cfg.vocab_path, cfg.max_length, context_length, number_sample)
