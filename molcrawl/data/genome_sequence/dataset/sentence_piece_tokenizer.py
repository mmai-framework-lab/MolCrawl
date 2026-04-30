from argparse import ArgumentParser
from pathlib import Path

from molcrawl.data.genome_sequence.utils.config import GenomeSequenceConfig


def train_tokenizer(output_dir, vocab_size, max_lines_per_file, input_sentence_size):
    import numpy as np
    import sentencepiece as spm

    """Train a tokenizer with sentence piece: https://github.com/google/sentencepiece"""
    path_dir = Path(output_dir) / "raw_files"
    files = list(path_dir.glob("*.raw"))
    files = np.random.permutation(files)

    spm.SentencePieceTrainer.train(
        input=",".join([str(f) for f in files[: int(input_sentence_size / max_lines_per_file) * 2]]),
        normalization_rule_name="identity",
        model_type="bpe",
        model_prefix=str(Path(output_dir) / "spm_tokenizer"),
        vocab_size=vocab_size,
        input_sentence_size=input_sentence_size,
        allow_whitespace_only_pieces=False,
        remove_extra_whitespaces=True,
        max_sentencepiece_length=50,
        split_by_whitespace=False,
        add_dummy_prefix=False,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    train_tokenizer(cfg.output_dir, cfg.vocab_size, cfg.max_lines_per_file, cfg.input_sentence_size)
