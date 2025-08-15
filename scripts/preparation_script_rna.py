"""
This script will download the cellxgene dataset.
There will be multiple directory generate in the output_dir provided in the configuration

- download_dir: Raw archive file downloaded from the cellxgene database
- extract: h5ad file extracted from the archives
- parquet_files: parquet files containing tokenized gene and expression values

You can call this script with the following command:

python scripts/preparation_script_rna.py assets/configs/rna.yaml
"""

from argparse import ArgumentParser
from pathlib import Path
import logging

import json
from datasets import load_dataset
import matplotlib.pyplot as plt

from rna.dataset.cellxgene.script.build_list import build_list
from rna.dataset.cellxgene.script.download import download
from rna.dataset.cellxgene.script.h5ad_to_loom import h5ad_to_loom
from rna.dataset.cellxgene.script.scgpt_tokenization import get_census_gene_vocab
from datasets.utils.logging import enable_progress_bar

from rna.dataset.tokenization import tokenize
from rna.utils.config import RnaConfig
from core.base import setup_logging

logger = logging.getLogger(__name__)
enable_progress_bar()

from config.paths import RNA_DATASET_DIR

def create_distribution_plot(data):
    plt.hist(data["num_tokens"], bins=200)
    plt.xlabel("Length of tokenized dataset")
    plt.title("Distribution of tokenized lengths")
    plt.tight_layout()
    plt.savefig("assets/img/rna_tokenized_lengths_dist.png")
    plt.close()
    logger.info(msg="Saved distribution of tokenized dataset lengths to assets/img/rna_tokenized_lengths_dist.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    setup_logging(Path(RNA_DATASET_DIR))

    build_list(RNA_DATASET_DIR, cfg.census_version)
    download(RNA_DATASET_DIR, cfg.census_version, cfg.num_worker, cfg.size_workload)
    h5ad_to_loom(RNA_DATASET_DIR)
    tokenize(RNA_DATASET_DIR)

    vocab = get_census_gene_vocab(cfg.census_version)
    vocab.save_json(Path(RNA_DATASET_DIR) / "gene_vocab.json")

    data = load_dataset(
        "parquet",
        data_dir=str(Path(RNA_DATASET_DIR) / "parquet_files"),
        split="train",
        cache_dir=str(Path(RNA_DATASET_DIR) / "hf_cache"),
    )

    logger.info(f"Number of sequence: {len(data)}")
    with open(Path(RNA_DATASET_DIR) / "gene_vocab.json") as file:
        vocab = json.load(file)
    logger.info(f"Size of the vocabulary: {len(vocab)}")

    logger.info(f"Number of tokens: {sum(data['token_count'])}")
    create_distribution_plot(data)
