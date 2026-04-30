"""https://github.com/MAGICS-LAB/DNABERT_2/issues/74"""

import os
from tqdm import tqdm

from typing import List
from pathlib import Path
from argparse import ArgumentParser


from molcrawl.data.genome_sequence.utils.config import GenomeSequenceConfig

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def read_file(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        return file.readlines()


def train_tokenizer(output_dir, vocab_size):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    import numpy as np

    path_dir = Path(output_dir) / "raw_files"

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        show_progress=True,
        vocab_size=vocab_size,
        min_frequency=2,
    )

    tokenizer.pre_tokenizer = Whitespace()

    files = [str(p) for p in path_dir.glob("*.raw")]
    total_size = 0
    filter_files = []
    np.random.permutation(files)
    for f in files:
        filter_files.append(f)
        total_size += os.path.getsize(f)
        if total_size > 10 * 10**9:
            break

    text = ""
    for f in tqdm(filter_files):
        with open(f, "r") as file:
            text += file.read()

    tokenizer.train(["assets/train.txt"], trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    # tokenizer.train_from_iterator(line_iterator, trainer)
    tokenizer.save(str(Path(output_dir) / "tokenizer.json"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    train_tokenizer(cfg.output_dir, cfg.vocab_size)
