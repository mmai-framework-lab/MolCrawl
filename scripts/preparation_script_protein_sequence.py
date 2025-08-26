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
    parser.add_argument("--force", action="store_true", help="Force re-download and reprocessing even if files exist")
    args = parser.parse_args()
    cfg = ProteinSequenceConfig.from_file(args.config).data_preparation
    cfg.max_lines_per_file = int(cfg.max_lines_per_file)

    setup_logging(PROTEIN_SEQUENCE_DIR)

    # マーカーファイル・出力ファイル
    download_marker = Path(PROTEIN_SEQUENCE_DIR) / "download_complete.marker"
    raw_marker = Path(PROTEIN_SEQUENCE_DIR) / "fasta_to_raw_complete.marker"
    parquet_marker = Path(PROTEIN_SEQUENCE_DIR) / "tokenize_to_parquet_complete.marker"
    processed_parquet = Path(PROTEIN_SEQUENCE_DIR) / "parquet_files" / "train.parquet"
    raw_files_dir = Path(PROTEIN_SEQUENCE_DIR) / "raw_files"
    fasta_dir = Path(PROTEIN_SEQUENCE_DIR) / "fasta_file"

    # 1. データダウンロード
    if not args.force and download_marker.exists():
        logger.info("Dataset download already completed. Skipping download step.")
    else:
        logger.info("Downloading protein dataset...")
        process_dataset(cfg.dataset, PROTEIN_SEQUENCE_DIR, cfg.num_worker, cfg.use_md5)
        download_marker.touch()
        logger.info("Download completed.")

    # 2. FASTA→raw変換
    if not args.force and raw_marker.exists() and raw_files_dir.exists() and any(raw_files_dir.glob("*.raw")):
        logger.info("FASTA to raw conversion already completed. Skipping conversion step.")
    else:
        os.makedirs(raw_files_dir, exist_ok=True)
        logger.info("Converting FASTA to raw...")
        fasta_to_raw_protein(cfg.dataset, PROTEIN_SEQUENCE_DIR, cfg.max_lines_per_file)
        raw_marker.touch()
        logger.info("FASTA to raw conversion completed.")

    # 3. トークナイズ・Parquet化
    if not args.force and parquet_marker.exists() and processed_parquet.exists():
        logger.info("Tokenization to Parquet already completed. Skipping tokenization step.")
    else:
        logger.info("Tokenizing to Parquet...")
        tokenize_to_parquet(PROTEIN_SEQUENCE_DIR, cfg.num_worker)
        parquet_marker.touch()
        logger.info("Tokenization to Parquet completed.")

    # 4. 統計・分布プロット
    logger.info("Loading Parquet dataset and generating statistics...")
    data = load_dataset(
        "parquet",
        data_dir=str(Path(PROTEIN_SEQUENCE_DIR) / "parquet_files"),
        cache_dir=str(Path(PROTEIN_SEQUENCE_DIR) / cfg.dataset / "hf_cache"),
    )
    logger.info(f"Number of sequence: {len(data['train'])}")
    tokenizer = EsmSequenceTokenizer()
    logger.info(f"Size of the vocabulary: {tokenizer.vocab_size}")
    logger.info(f"Number of tokens: {sum(data['train']['token_count'])}")
    plot_file = Path("assets/img/protein_sequence_tokenized_lengths_dist.png")
    if args.force or not plot_file.exists():
        if args.force:
            logger.info("Force option specified. Regenerating distribution plot...")
        logger.info("Creating distribution plot...")
        create_distribution_plot(data["train"])
    else:
        logger.info("Distribution plot already exists. Skipping plot generation.")
        logger.info("Use --force option to regenerate plot.")
