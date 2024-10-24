from typing import Iterator, List
from pathlib import Path
from argparse import ArgumentParser
import concurrent.futures

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
import numpy as np

from genome_sequence.utils.config import GenomeSequenceConfig


def yield_raw_sequences(raw_filepaths: Iterator[str], num_worker) -> Iterator[str]:
    """
    Reads sequences from a FASTA file and yields them one by one.

    Parameters:
    - fasta_filepath: Path to the input FASTA file.

    Yields:
    - A sequence string (without the header).
    """
    with concurrent.futures.ThreadPoolExecutor(num_worker) as executor:
        for lines in executor.map(read_file, raw_filepaths):
            for line in lines:
                yield line.strip()


def read_file(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        return file.readlines()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    path_dir = Path(cfg.output_dir) / "raw_files"

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=cfg.vocab_size, show_progress=True, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    pattern = r"[^A-Z]"
    tokenizer.pre_tokenizer = Split(pattern, behavior="removed")

    files = [str(p) for p in path_dir.glob("*.raw")]
    np.random.seed(42)
    np.random.permutation(files)
    files = files[:1200]

    line_iterator = yield_raw_sequences(files, cfg.num_worker)

    # tokenizer.train(files, trainer)
    tokenizer.train_from_iterator(line_iterator, trainer)
    tokenizer.save(str(Path(cfg.output_dir) / "tokenizer.json"))
