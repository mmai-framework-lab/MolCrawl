from argparse import ArgumentParser
from pathlib import Path
import logging

from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

from genome_sequence.dataset.refseq.download_refseq import download_refseq
from genome_sequence.dataset.refseq.fasta_to_raw import fasta_to_raw_genome

# from genome_sequence.dataset.train_tokenizer import train_tokenizer
from genome_sequence.dataset.sentence_piece_tokenizer import train_tokenizer
from genome_sequence.dataset.tokenizer import raw_to_parquet
from genome_sequence.utils.config import GenomeSequenceConfig
from core.base import setup_logging

from config.paths import GENOME_SEQUENCE_DIR

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
    setup_logging(GENOME_SEQUENCE_DIR)

    # process1
    logger.info("👉Process1 : Downloading RefSeq dataset...")
    download_refseq(GENOME_SEQUENCE_DIR, cfg.path_species, cfg.num_worker)

    # process2
    logger.info("👉Process2 : Converting FASTA to raw text...")
    logger.info(f" - Base Directory : {GENOME_SEQUENCE_DIR}")
    logger.info(f" - Number of Workers : {cfg.num_worker}")
    logger.info(f" - Max Lines per File : {cfg.max_lines_per_file}")
    fasta_to_raw_genome(GENOME_SEQUENCE_DIR, cfg.num_worker, cfg.max_lines_per_file)

    # process3
    logger.info("👉Process3 : Training tokenizer...")
    logger.info(f" - Base Directory : {GENOME_SEQUENCE_DIR}")
    logger.info(f" - vocab size : {cfg.vocab_size}")
    logger.info(f" - max lines per file : {cfg.max_lines_per_file}")
    logger.info(f" - input sentence size : {cfg.input_sentence_size}")
    train_tokenizer(GENOME_SEQUENCE_DIR, cfg.vocab_size, cfg.max_lines_per_file, cfg.input_sentence_size)

    # process4
    logger.info("👉Process4 : Converting raw text to Parquet...")
    #raw_to_parquet(GENOME_SEQUENCE_DIR)

    # process5
    logger.info("👉Process5 : Loading Parquet dataset...")
    data = load_dataset(
        "parquet",
        data_files=[str(Path(GENOME_SEQUENCE_DIR) / "parquet_files")],
        cache_dir=str(Path(GENOME_SEQUENCE_DIR) / "hf_cache"),
    )

    logger.info("👍Complete.")
    logger.info(f"Number of sequence: {len(data['train'])}")
    logger.info(f"Size of the vocabulary: {cfg.vocab_size}")
    logger.info(f"Number of tokens: {sum(data['train']['num_tokens'])}")
    create_distribution_plot(data)
