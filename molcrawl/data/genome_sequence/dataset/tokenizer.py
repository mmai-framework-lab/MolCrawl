from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from functools import partial

from molcrawl.data.genome_sequence.utils.config import GenomeSequenceConfig


def tokenize_function(examples, tokenizer):
    # encoded_sequence = tokenizer.encode(examples["text"])["input_ids"] # autoTokenizer case
    encoded_sequence = tokenizer.encode(examples["text"])
    return {"input_ids": encoded_sequence, "num_tokens": len(encoded_sequence)}


# def raw_to_parquet(output_dir):
#    data = load_dataset(
#        "text",
#        data_dir=str(Path(output_dir) / "raw_files"),
#        cache_dir=str(Path(output_dir) / "hf_cache"),
#        split="train",
#    )

#    # tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
#    tokenizer = spm.SentencePieceProcessor(model_file=str(Path(output_dir) / "spm_tokenizer.model"))

#    tokenized_datasets = data.map(
#        partial(tokenize_function, tokenizer=tokenizer),
#        batched=False,
#        remove_columns=["text"],
#    )

#    tokenized_datasets.to_parquet(str(Path(output_dir) / "parquet_files"))


def raw_to_parquet(output_dir, num_proc=None, batch_size=None, shard_rows=200_000):
    """Tokenize the raw_files/ corpus with SentencePiece and write parquet shards.

    The output ``parquet_files/`` was historically a single .parquet file.
    PyArrow uses int32 offsets for ``list<int32>`` columns, which overflow at
    ~2 GB of cumulative list bytes — fine for the legacy genome corpus
    (~85 GB raw → ~26 GB parquet) but it fails on the ambiguity-aware v2
    corpus (~111 GB raw) with::

        offset overflow while concatenating arrays, consider casting input
        from ``list<item: int32>`` to ``large_list<item: int32>`` first.

    To avoid the limit and to keep ``prepare_gpt2.py`` simple, we now write
    ``parquet_files/`` as a *directory* of contiguous shards. The downstream
    loader globs ``*.parquet`` under it. A legacy single-file
    ``parquet_files`` (v1) is still readable by the loader as-is.
    """
    from datasets import load_dataset
    import sentencepiece as spm

    data = load_dataset(
        "text",
        data_dir=str(Path(output_dir) / "raw_files"),
        cache_dir=str(Path(output_dir) / "hf_cache"),
        split="train",
    )

    tokenizer = spm.SentencePieceProcessor(model_file=str(Path(output_dir) / "spm_tokenizer.model"))

    def batched_tokenize(batch, tokenizer):
        texts = batch["text"]
        outputs = [tokenize_function({"text": t}, tokenizer=tokenizer) for t in texts]
        keys = outputs[0].keys()
        result = {k: [o[k] for o in outputs] for k in keys}
        return result

    tokenized_datasets = data.map(
        partial(batched_tokenize, tokenizer=tokenizer),
        batched=True,
        batch_size=batch_size or 512,
        num_proc=num_proc,
        remove_columns=["text"],
    )

    parquet_path = Path(output_dir) / "parquet_files"
    # If a stale single-file v1 artifact lives at the target path, remove it
    # so the directory layout can be created without conflict.
    if parquet_path.exists() and not parquet_path.is_dir():
        parquet_path.unlink()
    parquet_path.mkdir(parents=True, exist_ok=True)

    n_rows = len(tokenized_datasets)
    num_shards = max(1, (n_rows + shard_rows - 1) // shard_rows)
    print(f"Writing {n_rows:,} rows as {num_shards} parquet shards under {parquet_path}/")
    for i in range(num_shards):
        shard = tokenized_datasets.shard(num_shards=num_shards, index=i, contiguous=True)
        shard_path = parquet_path / f"shard-{i:05d}-of-{num_shards:05d}.parquet"
        shard.to_parquet(str(shard_path))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    raw_to_parquet(cfg.output_dir)
