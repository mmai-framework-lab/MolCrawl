from argparse import ArgumentParser
import os
import sys
from pathlib import Path
from functools import partial

# プロジェクトルートのsrcディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# データセットキャッシュ設定を読み込み（configs/cache.yamlから）
try:
    # 任意のキャッシュ設定。存在しない環境でも学習は継続できる。
    from utils.cache_config import setup_cache_env
except ModuleNotFoundError:
    setup_cache_env = None

if setup_cache_env is not None:
    setup_cache_env()
else:
    # cache_configが無い環境でも動作は可能
    print("WARNING: utils.cache_config not found. Continuing without cache setup.")

# from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict

from rna.utils.config import RnaConfig


def concatenate_texts(examples, eos_token_id):
    concatenated_ids = []
    for input_ids in examples["token"]:
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


def tokenize_batch_dataset(output_dir, context_length, number_sample):
    data = (
        load_dataset(
            "parquet",
            data_dir=str(Path(output_dir) / "parquet_files"),
            cache_dir=str(Path(output_dir) / "hf_cache"),
            split="train",
        ).shuffle()
        # .select(range(number_sample))
    )

    tokenized_datasets = data.train_test_split(test_size=0.2)
    valid_test_split = tokenized_datasets["test"].train_test_split(test_size=0.5)
    tokenized_datasets = DatasetDict(
        {"train": tokenized_datasets["train"], "valid": valid_test_split["train"], "test": valid_test_split["test"]}
    )

    concatenated_dataset = tokenized_datasets.map(
        partial(concatenate_texts, eos_token_id=0),
        batched=True,
        batch_size=context_length * 100,
        remove_columns=["token", "token_count"],
    )

    chunked_dataset = concatenated_dataset.map(
        partial(create_chunks, context_length=context_length), batched=True, batch_size=context_length * 10
    )

    path_dataset = str(Path(output_dir) / "training_ready_hf_dataset")
    print(f"Saving dataset to: {path_dataset}. Match this path to the train_gpt2_config.py->dataset_dir parameter.")
    chunked_dataset.save_to_disk(path_dataset)


if __name__ == "__main__":
    number_sample = 50000
    context_length = 1024

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    tokenize_batch_dataset(cfg.output_dir, context_length, number_sample)
