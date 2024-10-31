from argparse import ArgumentParser
from pathlib import Path
from functools import partial

from tokenizers import Tokenizer
from datasets import load_dataset

from genome_sequence.utils.config import GenomeSequenceConfig


def tokenize_function(examples, tokenizer):
    encoded_sequence = tokenizer.encode(examples["text"]).ids
    return {"input_ids": encoded_sequence, "num_tokens": len(encoded_sequence)}


def raw_to_parquet(output_dir):
    data = load_dataset(
        "text", data_dir=str(Path(output_dir) / "raw_files"), cache_dir=str(Path(output_dir) / "hf_cache"), split="train"
    ).select(range(1000))

    tokenizer = Tokenizer.from_file(str(Path(output_dir) / "tokenizer.json"))

    tokenized_datasets = data.map(
        partial(tokenize_function, tokenizer=tokenizer),
        batched=False,
        remove_columns=["text"],
    )

    tokenized_datasets.to_parquet(str(Path(output_dir) / "parquet_files"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    raw_to_parquet(cfg.output_dir)
