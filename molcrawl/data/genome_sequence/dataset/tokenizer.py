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


def raw_to_parquet(output_dir, num_proc=None, batch_size=None):
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
    tokenized_datasets.to_parquet(str(Path(output_dir) / "parquet_files"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    raw_to_parquet(cfg.output_dir)
