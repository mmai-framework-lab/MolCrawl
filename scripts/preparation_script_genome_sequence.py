from argparse import ArgumentParser
from pathlib import Path
import logging

from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

from genome_sequence.dataset.refseq.download_refseq import download_refseq
from genome_sequence.dataset.refseq.fasta_to_raw import fasta_to_raw

# from genome_sequence.dataset.train_tokenizer import train_tokenizer
from genome_sequence.dataset.sentence_piece_tokenizer import train_tokenizer
from genome_sequence.dataset.tokenizer import raw_to_parquet
from genome_sequence.utils.config import GenomeSequenceConfig
from core.base import setup_logging

logger = logging.getLogger(__name__)


def create_distribution_plot(data):
    plt.hist(data["train"]["num_tokens"], bins=np.arange(0, 200, 1))
    plt.xlabel("Length of tokenized dataset")
    plt.title("Distribution of tokenized lengths")
    plt.savefig("assets/img/genome_sequence_tokenized_lengths_dist.png")
    plt.close()
    logger.info(msg="Saved distribution of tokenized dataset lengths to assets/img/genome_sequence_tokenized_lengths_dist.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    setup_logging(cfg.output_dir)

    download_refseq(cfg.output_dir, cfg.path_species, cfg.num_worker)
    fasta_to_raw(cfg.output_dir, cfg.num_worker, cfg.max_lines_per_file)
    train_tokenizer(cfg.output_dir, cfg.vocab_size, cfg.max_lines_per_file, cfg.input_sentence_size)
    raw_to_parquet(cfg.output_dir)

    data = load_dataset(
        "parquet",
        data_files=[str(Path(cfg.output_dir) / "parquet_files")],
        cache_dir=str(Path(cfg.output_dir) / "hf_cache"),
    )

    logger.info(f"Number of sequence: {len(data['train'])}")
    logger.info(f"Size of the vocabulary: {cfg.vocab_size}")
    logger.info(f"Number of tokens: {sum(data['train']['num_tokens'])}")
    create_distribution_plot(data)
