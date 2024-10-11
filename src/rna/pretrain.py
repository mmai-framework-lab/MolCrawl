from pathlib import Path
import logging

from datasets import load_dataset
from scgpt.tokenizer import GeneVocab
from scgpt import DataCollator as ScGPTDataCollator

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parquet_dir = ""
    vocab_path = ""
    valid_size_or_ratio = 0.1
    max_seq_len = 600
    input_style = "binned"
    mask_ratio = 0.4
    n_bins = 51
    trunc_by_sample = True

    cache_dir = Path(parquet_dir).parent / "hf_cache"

    vocab = GeneVocab.from_file(Path(vocab_path))
    raw_dataset = load_dataset(
        "parquet",
        data_dir=parquet_dir,
        split="train",
        cache_dir=str(cache_dir),
    )
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # # Data processing
    # convert format to return torch.tensor
    raw_dataset = raw_dataset.with_format("torch")

    # split train and validation,
    raw_dataset = raw_dataset.train_test_split(test_size=valid_size_or_ratio, shuffle=True)
    train_dataset = raw_dataset["train"]
    valid_dataset = raw_dataset["test"]
    logger.info(f"train set number of samples: {len(train_dataset)}, ")
    logger.info(f"valid set number of samples: {len(valid_dataset)}, ")

    mask_value = -1
    pad_value = -2

    # data collator for online padding and sampling
    # make separate two types of input and output
    collator = ScGPTDataCollator(
        do_padding=True if max_seq_len is not None else False,
        pad_token_id=vocab[pad_token],
        pad_value=pad_value,
        do_mlm=True,
        do_binning=True if input_style == "binned" else False,
        mlm_probability=mask_ratio,
        mask_value=mask_value,
        max_length=max_seq_len,
        sampling=trunc_by_sample,
    )
