from typing import List
from pathlib import Path
from argparse import ArgumentParser

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import numpy as np

from genome_sequence.utils.config import GenomeSequenceConfig


def read_file(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        return file.readlines()


def train_tokenizer(output_dir, vocab_size):
    path_dir = Path(output_dir) / "raw_files"

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        max_token_length=10,
    )

    files = [str(p) for p in path_dir.glob("*.raw")]
    np.random.seed(42)
    np.random.permutation(files)
    files = files[:35]
    tokenizer.train(files, trainer)
    # tokenizer.train_from_iterator(line_iterator, trainer)
    tokenizer.save(str(Path(output_dir) / "tokenizer.json"))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    train_tokenizer(cfg.output_dir, cfg.vocab_size)
