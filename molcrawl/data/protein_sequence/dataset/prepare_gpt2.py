from __future__ import annotations

import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

# Opt-in parallelism for the per-sequence tokenize map. Default 1 keeps the
# original single-process behaviour; set PROTEIN_TOKENIZE_NPROC>1 to parallelize
# the char-level ESM tokenization of the (tens of millions of) UniRef sequences.
_PROT_TOKENIZE_NPROC = max(1, int(os.environ.get("PROTEIN_TOKENIZE_NPROC", "1") or "1"))

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


def pack_split_stream(tokenized_split, eos_token_id: int, context_length: int):
    """Return a zero-arg generator that packs one split into fixed-length blocks.

    Output is byte-identical to a single global concatenate+chunk (the old
    ``batch_size=-1`` path): the split is one continuous EOS-joined token stream
    sliced into ``context_length`` blocks with a single trailing remainder
    (< context_length) dropped. Memory is O(context_length) — it never
    materialises the whole corpus, so it is safe on full UniRef50.
    """

    def gen():
        buf: List[int] = []
        pos = 0
        for row in tokenized_split:
            buf.extend(row["input_ids"])
            buf.append(eos_token_id)
            while len(buf) - pos >= context_length:
                yield {"input_ids": buf[pos : pos + context_length]}
                pos += context_length
            if pos >= 1_000_000:  # compact occasionally to bound memory
                del buf[:pos]
                pos = 0
        # trailing remainder (< context_length) is dropped — matches batch_size=-1

    return gen


def tokenize_batch_dataset(path_output: Path, context_length: int, number_sample: Optional[int]) -> None:
    """Tokenize and chunk the protein corpus into fixed-length samples.

    ``number_sample`` semantics:
    - ``None`` (or 0 / negative): use every sequence from the raw files
      (recommended for full pretraining on UniRef50's ~60M rows).
    - positive int: sample the first ``number_sample`` rows after
      shuffling (useful for smoke tests; reproduces the old hard-coded
      behaviour when set to 50000).
    """
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
    data = load_dataset(
        "text",
        data_files={"train": [str(p) for p in raw_files]},
        split="train",
    ).shuffle(seed=PREP_SHUFFLE_SEED)

    if number_sample is not None and number_sample > 0:
        data = data.select(range(min(number_sample, len(data))))
    raw_datasets = data.train_test_split(test_size=0.2, seed=PREP_SHUFFLE_SEED)
    valid_test_split = raw_datasets["test"].train_test_split(test_size=0.5, seed=PREP_SHUFFLE_SEED)
    raw_datasets = DatasetDict(
        {"train": raw_datasets["train"], "valid": valid_test_split["train"], "test": valid_test_split["test"]}
    )

    tokenizer: EsmSequenceTokenizer = EsmSequenceTokenizer()

    tokenized_datasets = raw_datasets.map(
        partial(tokenize_function, tokenizer=tokenizer),
        batched=True,
        remove_columns=["text"],
        num_proc=_PROT_TOKENIZE_NPROC,
    )

    # Pack via a streaming carry-over generator (see pack_split_stream): the old
    # batch_size=-1 concat+chunk OOMs on full UniRef50, and any bounded batch_size
    # would silently change packing at every batch boundary. This is byte-identical
    # to batch_size=-1 with O(context_length) memory.
    from datasets import Dataset

    packed = DatasetDict(
        {
            split: Dataset.from_generator(
                pack_split_stream(tokenized_datasets[split], tokenizer.eos_token_id, context_length)
            )
            for split in tokenized_datasets
        }
    )

    path_dataset: str = str(path_output / f"training_ready_hf_dataset{PREP_OUT_SUFFIX}")
    print(f"Saving dataset to: {path_dataset}. Match this path to the train_gpt2_config.py->dataset_dir parameter.")
    packed.save_to_disk(path_dataset)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = ProteinSequenceConfig.from_file(args.config).data_preparation

    # Read with backward-compatible defaults:
    #   number_sample=None  => use every sequence from the raw files
    #   context_length=1024 => preserves the prior hard-coded value
    number_sample: Optional[int] = getattr(cfg, "number_sample", None)
    context_length: int = getattr(cfg, "context_length", 1024)

    output_dir: Path = Path(cfg.output_dir)
    tokenize_batch_dataset(output_dir, context_length, number_sample)
