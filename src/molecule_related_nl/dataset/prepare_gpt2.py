from functools import partial
from argparse import ArgumentParser
import os
import sys
from pathlib import Path

# プロジェクトルートのsrcディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets import DatasetDict
import numpy as np

from molecule_related_nl.utils.tokenizer import MoleculeNatLangTokenizer
from molecule_related_nl.utils.config import MoleculeNLConfig
from molecule_related_nl.utils.general import read_dataset


def concatenate_texts(examples, eos_token_id):
    concatenated_ids = []
    for input_ids, output_ids in zip(examples["input_ids"], examples["output_ids"]):
        concatenated_ids.extend(input_ids + output_ids)
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


def tokenize_batch_dataset(parquet_path, context_length, number_sample):

    tokenize_dataset = DatasetDict(read_dataset(parquet_path))
    
    # Handle validation/valid split naming
    if "validation" in tokenize_dataset and "valid" not in tokenize_dataset:
        tokenize_dataset["valid"] = tokenize_dataset["validation"]
        del tokenize_dataset["validation"]
    elif "valid" not in tokenize_dataset and "validation" not in tokenize_dataset:
        raise KeyError("Neither 'valid' nor 'validation' split found in dataset")

    tokenize_dataset["train"] = tokenize_dataset["train"].select(
        np.random.choice(len(tokenize_dataset["train"]), int(number_sample * 0.8), replace=False)
    )
    tokenize_dataset["valid"] = tokenize_dataset["valid"].select(
        np.random.choice(len(tokenize_dataset["valid"]), int(number_sample * 0.1), replace=False)
    )
    tokenize_dataset["test"] = tokenize_dataset["test"].select(
        np.random.choice(len(tokenize_dataset["test"]), int(number_sample * 0.1), replace=False)
    )

    tokenizer = MoleculeNatLangTokenizer()

    concatenated_dataset = tokenize_dataset.map(
        partial(concatenate_texts, eos_token_id=tokenizer.tokenizer.eos_token_id),
        batched=True,
        batch_size=-1,
        remove_columns=tokenize_dataset["train"].column_names,
    )

    chunked_dataset = concatenated_dataset.map(
        partial(create_chunks, context_length=context_length),
        batched=True,
        batch_size=-1,
    )

    path_dataset = str(Path(parquet_path).parent / "training_ready_hf_dataset")
    print(f"Saving dataset to: {path_dataset}. Match this path to the train_gpt2_config.py->dataset_dir parameter.")
    chunked_dataset.save_to_disk(path_dataset)


if __name__ == "__main__":
    number_sample = 50000
    context_length = 1024

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = MoleculeNLConfig.from_file(args.config).data_preparation
    
    # 相対パスを絶対パスに変換
    from config.paths import PROJECT_ROOT, LEARNING_SOURCE_DIR
    save_path = os.path.join(PROJECT_ROOT, LEARNING_SOURCE_DIR, cfg.save_path)

    tokenize_batch_dataset(save_path, context_length, number_sample)
