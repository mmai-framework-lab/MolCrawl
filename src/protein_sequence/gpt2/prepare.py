from functools import partial
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset, DatasetDict

from protein_sequence.dataset.tokenizer import EsmSequenceTokenizer
from protein_sequence.utils.configs import ProteinSequenceConfig


def tokenize_function(examples, tokenizer):
    return {
        "input_ids": tokenizer(
            examples["text"],
            truncation=False,
            add_special_tokens=False,  # We'll add special tokens manually
        )["input_ids"]
    }


def concatenate_texts(examples, eos_token_id):
    concatenated_ids = []
    for input_ids in examples["input_ids"]:
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


def tokenize_batch_dataset(path_output, context_length, number_sample):
    data = (
        load_dataset(
            "text",
            data_dir=str(Path(path_output) / "raw_files"),
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

    tokenizer = EsmSequenceTokenizer()

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

    path_dataset = str(path_output / "training_ready_hf_dataset")
    chunked_dataset.save_to_disk(path_dataset)


if __name__ == "__main__":
    number_sample = 50000
    context_length = 1024

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = ProteinSequenceConfig.from_file(args.config).data_preparation

    tokenize_batch_dataset(cfg.dataset, context_length, number_sample)
