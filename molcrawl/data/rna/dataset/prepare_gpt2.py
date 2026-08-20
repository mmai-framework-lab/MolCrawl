import os
from argparse import ArgumentParser
from pathlib import Path
from functools import partial

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

# from transformers import AutoTokenizer
from molcrawl.data.rna.utils.config import RnaConfig

# Opt-in parallelism for the concatenate/chunk maps. Default 1 keeps the
# original single-process behaviour and byte-identical packing. On machines
# with a tight walltime, set RNA_TOKENIZE_NPROC>1 to parallelize (packing then
# differs only at shard boundaries, which is fine for LM pretraining).
_RNA_TOKENIZE_NPROC = max(1, int(os.environ.get("RNA_TOKENIZE_NPROC", "1") or "1"))

# Seed for the shuffle and the train/valid/test draw. Both were unseeded, so neither
# the split membership nor the block contents could be reproduced from a rebuild,
# which left the seed fixation added for the training configs stopping at the data
# boundary.
PREP_SHUFFLE_SEED = 42

# Appended to the output directory name. Empty writes the production path the
# configs point at; set it for any rebuild that is not meant to replace the data the
# finished runs used. Seeding the draw above changes split membership, so a rebuild
# is not interchangeable with the current data and must not overwrite it.
PREP_OUT_SUFFIX = os.environ.get("PREP_OUT_SUFFIX", "")

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
    """Tokenize and chunk the RNA corpus into fixed-length samples.

    ``number_sample`` semantics:
    - ``None`` (or 0 / negative): use the entire parquet corpus (recommended
      for full pretraining).
    - positive int: sample the first ``number_sample`` rows after shuffling
      (useful for smoke tests; reproduces the old hard-coded behaviour when
      set to 50000).
    """
    from datasets import DatasetDict, load_dataset

    data = load_dataset(
        "parquet",
        data_dir=str(Path(output_dir) / "parquet_files"),
        cache_dir=str(Path(output_dir) / "hf_cache"),
        split="train",
    ).shuffle(seed=PREP_SHUFFLE_SEED)

    if number_sample is not None and number_sample > 0:
        data = data.select(range(min(number_sample, len(data))))

    tokenized_datasets = data.train_test_split(test_size=0.2, seed=PREP_SHUFFLE_SEED)
    valid_test_split = tokenized_datasets["test"].train_test_split(test_size=0.5, seed=PREP_SHUFFLE_SEED)
    tokenized_datasets = DatasetDict(
        {"train": tokenized_datasets["train"], "valid": valid_test_split["train"], "test": valid_test_split["test"]}
    )

    if _RNA_TOKENIZE_NPROC > 1:
        print(f"Tokenize maps using num_proc={_RNA_TOKENIZE_NPROC}")
    concatenated_dataset = tokenized_datasets.map(
        partial(concatenate_texts, eos_token_id=0),
        batched=True,
        batch_size=context_length * 100,
        remove_columns=["token", "token_count"],
        num_proc=_RNA_TOKENIZE_NPROC,
    )

    chunked_dataset = concatenated_dataset.map(
        partial(create_chunks, context_length=context_length),
        batched=True,
        batch_size=context_length * 10,
        num_proc=_RNA_TOKENIZE_NPROC,
    )

    path_dataset = str(Path(output_dir) / f"training_ready_hf_dataset{PREP_OUT_SUFFIX}")
    print(f"Saving dataset to: {path_dataset}. Match this path to the train_gpt2_config.py->dataset_dir parameter.")
    chunked_dataset.save_to_disk(path_dataset)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    # Read with backward-compatible defaults:
    #   number_sample=None  => use the full parquet corpus
    #   context_length=1024 => preserves the prior hard-coded value
    number_sample = getattr(cfg, "number_sample", None)
    context_length = getattr(cfg, "context_length", 1024)

    tokenize_batch_dataset(cfg.output_dir, context_length, number_sample)
