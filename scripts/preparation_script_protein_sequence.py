"""
This script will download on of the Uniprot dataset base on the name in the config.

The output will be the a subdir of the output_dir containing a dataset name directory (ex uniprot_50) containing the rest of the file:

- Archive file, for uniprot a archive dir will be creating containing all the files
- A fasta file extracted from the archive, for uniprot a fasta_file directory will be created containing all the file.
- A raw_files directory containing multiple file with one protein sequence per line.
- A parquet_files directory, containing two column parquet file tokenized sequence ("token") and the number of ("token_count")
- A token_counts.pkl file which contains a list of int corresponding to token_count for computing statistics of the dataset.

You can run this script with the following command:

python scripts/preparation_script_protein_sequence.py assets/configs/protein_sequence.yaml

"""

from argparse import ArgumentParser
from pathlib import Path
import logging, os

from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

from protein_sequence.dataset.uniprot.uniprot_download import process_dataset
from protein_sequence.dataset.uniprot.fasta_to_raw import fasta_to_raw_protein
from protein_sequence.dataset.tokenizer import tokenize_to_parquet
from protein_sequence.dataset.tokenizer import EsmSequenceTokenizer
from protein_sequence.utils.configs import ProteinSequenceConfig
from core.base import setup_logging

from config.paths import PROTEIN_SEQUENCE_DIR

logger = logging.getLogger(__name__)


def create_distribution_plot(data):
    plt.hist(data["token_count"], bins=np.arange(0, 1000, 1))
    plt.xlabel("Length of tokenized dataset")
    plt.title("Distribution of tokenized lengths (cut at 1000)")
    plt.savefig("assets/img/protein_sequence_tokenized_lengths_dist.png")
    plt.close()
    logger.info(msg="Saved distribution of tokenized dataset lengths to assets/img/protein_sequence_tokenized_lengths_dist.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = ProteinSequenceConfig.from_file(args.config).data_preparation
    cfg.max_lines_per_file = int(cfg.max_lines_per_file)

    setup_logging(PROTEIN_SEQUENCE_DIR)

    process_dataset(cfg.dataset, PROTEIN_SEQUENCE_DIR, cfg.num_worker, cfg.use_md5)
    os.makedirs(Path(PROTEIN_SEQUENCE_DIR) / "raw_files", exist_ok=True)
    fasta_to_raw_protein(cfg.dataset, PROTEIN_SEQUENCE_DIR, cfg.max_lines_per_file)
    tokenize_to_parquet(PROTEIN_SEQUENCE_DIR, cfg.num_worker)

    data = load_dataset(
        "parquet",
        data_dir=str(Path(PROTEIN_SEQUENCE_DIR) / "parquet_files"),
        cache_dir=str(Path(PROTEIN_SEQUENCE_DIR) / cfg.dataset / "hf_cache"),
    )
    logger.info(f"Number of sequence: {len(data['train'])}")
    tokenizer = EsmSequenceTokenizer()
    logger.info(f"Size of the vocabulary: {tokenizer.vocab_size}")
    logger.info(f"Number of tokens: {sum(data['train']['token_count'])}")
    create_distribution_plot(data["train"])
