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

import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

# プロジェクトルートのsrcディレクトリをパスに追加

from molcrawl.config.paths import PROTEIN_SEQUENCE_DIR
from molcrawl.core.base import setup_logging
from molcrawl.protein_sequence.dataset.tokenizer import EsmSequenceTokenizer, tokenize_to_parquet
from molcrawl.protein_sequence.dataset.uniprot.fasta_to_raw import fasta_to_raw_protein
from molcrawl.protein_sequence.dataset.uniprot.uniprot_download import process_dataset
from molcrawl.protein_sequence.utils.configs import ProteinSequenceConfig

logger = logging.getLogger(__name__)


def create_distribution_plot(data):
    """Create and save distribution plot for tokenized sequence lengths"""
    try:
        from utils.image_manager import get_image_path

        plt.hist(data["token_count"], bins=np.arange(0, 1000, 1))
        plt.xlabel("Length of tokenized dataset")
        plt.title("Distribution of tokenized lengths (cut at 1000)")

        image_path = get_image_path("protein_sequence", "protein_sequence_tokenized_lengths_dist.png")
        plt.savefig(image_path)
        plt.close()
        logger.info(f"Saved distribution of tokenized dataset lengths to {image_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create distribution plot: {e}")
        return False


def check_progress_status(base_dir):
    """Check the progress status of all processing steps

    Args:
        base_dir (str): Base directory for protein sequence data

    Returns:
        bool: True if all steps are completed, False otherwise
    """
    # 各処理段階のマーカーファイルパス
    download_marker = Path(base_dir) / "download_complete.marker"
    raw_marker = Path(base_dir) / "fasta_to_raw_complete.marker"
    parquet_marker = Path(base_dir) / "tokenize_to_parquet_complete.marker"

    # 出力ディレクトリとファイルの存在確認用パス
    raw_files_dir = Path(base_dir) / "raw_files"
    processed_parquet = Path(base_dir) / "parquet_files" / "train.parquet"

    # 進捗状況の確認
    logger.info("=== Protein Sequence Dataset Preparation Progress ===")
    steps_completed = 0
    total_steps = 3

    if download_marker.exists():
        logger.info("✓ Step 1/3: Uniprot dataset download - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 1/3: Uniprot dataset download - PENDING")

    if raw_marker.exists() and raw_files_dir.exists() and any(raw_files_dir.glob("*.raw")):
        logger.info("✓ Step 2/3: FASTA to raw conversion - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 2/3: FASTA to raw conversion - PENDING")

    if parquet_marker.exists() and processed_parquet.exists():
        logger.info("✓ Step 3/3: Tokenization to Parquet - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 3/3: Tokenization to Parquet - PENDING")

    logger.info(f"Progress: {steps_completed}/{total_steps} steps completed")
    logger.info("====================================================")

    return steps_completed == total_steps


def process1_download_uniprot(base_dir, dataset, num_worker, use_md5, force=False):
    """Process 1: Download Uniprot dataset

    Args:
        base_dir (str): Base directory for protein sequence data
        dataset (str): Dataset name (e.g., uniprot_50)
        num_worker (int): Number of workers for parallel processing
        use_md5 (bool): Whether to use MD5 verification
        force (bool): Force re-download even if already completed

    Returns:
        bool: True if successful, False otherwise
    """
    download_marker = Path(base_dir) / "download_complete.marker"

    if not force and download_marker.exists():
        logger.info("👉Process1 : Uniprot dataset download already completed. Skipping...")
        logger.info("Use --force option to re-download.")
        return True

    try:
        if force:
            logger.info("👉Process1 : Force option specified. Re-downloading Uniprot dataset...")
        else:
            logger.info("👉Process1 : Downloading Uniprot dataset...")

        logger.info(f" - Dataset: {dataset}")
        logger.info(f" - Base Directory: {base_dir}")
        logger.info(f" - Number of Workers: {num_worker}")
        logger.info(f" - Use MD5 Verification: {use_md5}")

        process_dataset(dataset, base_dir, num_worker, use_md5)
        download_marker.touch()
        logger.info("Uniprot dataset download completed.")
        return True

    except Exception as e:
        logger.error(f"Uniprot dataset download failed: {e}")
        return False


def process2_fasta_to_raw(base_dir, dataset, max_lines_per_file, force=False):
    """Process 2: Convert FASTA files to raw text format

    Args:
        base_dir (str): Base directory for protein sequence data
        dataset (str): Dataset name
        max_lines_per_file (int): Maximum lines per output file
        force (bool): Force reconversion even if already completed

    Returns:
        bool: True if successful, False otherwise
    """
    raw_marker = Path(base_dir) / "fasta_to_raw_complete.marker"
    raw_files_dir = Path(base_dir) / "raw_files"

    if not force and raw_marker.exists() and raw_files_dir.exists() and any(raw_files_dir.glob("*.raw")):
        logger.info("👉Process2 : FASTA to raw conversion already completed. Skipping...")
        logger.info("Use --force option to reconvert.")
        return True

    try:
        if force:
            logger.info("👉Process2 : Force option specified. Reconverting FASTA to raw text...")
        else:
            logger.info("👉Process2 : Converting FASTA to raw text...")

        logger.info(f" - Dataset: {dataset}")
        logger.info(f" - Base Directory: {base_dir}")
        logger.info(f" - Max Lines per File: {max_lines_per_file}")

        # 出力ディレクトリを作成
        os.makedirs(raw_files_dir, exist_ok=True)

        fasta_to_raw_protein(dataset, base_dir, max_lines_per_file)
        raw_marker.touch()
        logger.info("FASTA to raw conversion completed.")
        return True

    except Exception as e:
        logger.error(f"FASTA to raw conversion failed: {e}")
        return False


def process3_tokenize_to_parquet(base_dir, num_worker, force=False):
    """Process 3: Tokenize raw files and convert to Parquet format

    Args:
        base_dir (str): Base directory for protein sequence data
        num_worker (int): Number of workers for parallel processing
        force (bool): Force retokenization even if already completed

    Returns:
        bool: True if successful, False otherwise
    """
    parquet_marker = Path(base_dir) / "tokenize_to_parquet_complete.marker"
    processed_parquet = Path(base_dir) / "parquet_files" / "train.parquet"

    if not force and parquet_marker.exists() and processed_parquet.exists():
        logger.info("👉Process3 : Tokenization to Parquet already completed. Skipping...")
        logger.info("Use --force option to retokenize.")
        return True

    try:
        if force:
            logger.info("👉Process3 : Force option specified. Retokenizing to Parquet...")
        else:
            logger.info("👉Process3 : Tokenizing to Parquet...")

        logger.info(f" - Base Directory: {base_dir}")
        logger.info(f" - Number of Workers: {num_worker}")

        tokenize_to_parquet(base_dir, num_worker)
        parquet_marker.touch()
        logger.info("Tokenization to Parquet completed.")
        return True

    except Exception as e:
        logger.error(f"Tokenization to Parquet failed: {e}")
        return False


def process4_generate_statistics(base_dir, dataset, force=False):
    """Process 4: Generate statistics and distribution plots

    Args:
        base_dir (str): Base directory for protein sequence data
        dataset (str): Dataset name for cache directory
        force (bool): Force regeneration even if already exists

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("👉Process4 : Loading Parquet dataset and generating statistics...")

    try:
        data = load_dataset(
            "parquet",
            data_dir=str(Path(base_dir) / "parquet_files"),
            cache_dir=str(Path(base_dir) / dataset / "hf_cache"),
        )

        logger.info("👍Dataset loaded successfully.")
        logger.info(f"Number of sequence: {len(data['train'])}")

        # トークナイザーの語彙サイズを取得
        tokenizer = EsmSequenceTokenizer()
        logger.info(f"Size of the vocabulary: {tokenizer.vocab_size}")
        logger.info(f"Number of tokens: {sum(data['train']['token_count'])}")

        # 分布プロットの生成（forceオプションまたはプロットが存在しない場合のみ）
        from utils.image_manager import get_image_path

        plot_file = Path(get_image_path("protein_sequence", "protein_sequence_tokenized_lengths_dist.png"))
        if force or not plot_file.exists():
            if force:
                logger.info("Force option specified. Regenerating distribution plot...")
            logger.info("Creating distribution plot...")

            if not create_distribution_plot(data["train"]):
                logger.warning("Distribution plot generation failed, but continuing...")
        else:
            logger.info("Distribution plot already exists. Skipping plot generation.")
            logger.info("Use --force option to regenerate plot.")

        return True

    except Exception as e:
        logger.error(f"Failed to load or process final dataset: {e}")
        return False


def main():
    """Main function to orchestrate the protein sequence dataset preparation"""
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and reprocessing even if files exist",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip statistics generation and plotting",
    )
    args = parser.parse_args()

    cfg = ProteinSequenceConfig.from_file(args.config).data_preparation
    cfg.max_lines_per_file = int(cfg.max_lines_per_file)

    setup_logging(PROTEIN_SEQUENCE_DIR)

    # 進捗状況の確認
    all_completed = check_progress_status(PROTEIN_SEQUENCE_DIR)

    if all_completed and not args.force:
        logger.info("All processing steps are already completed!")
        logger.info("Use --force option if you want to reprocess everything.")
        if not args.skip_stats:
            # 統計情報のみ生成
            process4_generate_statistics(PROTEIN_SEQUENCE_DIR, cfg.dataset, args.force)
        return

    # 各プロセスを順次実行
    success = True

    # Process 1: Download Uniprot dataset
    success &= process1_download_uniprot(PROTEIN_SEQUENCE_DIR, cfg.dataset, cfg.num_worker, cfg.use_md5, args.force)

    if not success:
        logger.error("Process 1 failed. Stopping execution.")
        exit(1)

    # Process 2: Convert FASTA to raw text
    success &= process2_fasta_to_raw(PROTEIN_SEQUENCE_DIR, cfg.dataset, cfg.max_lines_per_file, args.force)

    if not success:
        logger.error("Process 2 failed. Stopping execution.")
        exit(1)

    # Process 3: Tokenize to Parquet
    success &= process3_tokenize_to_parquet(PROTEIN_SEQUENCE_DIR, cfg.num_worker, args.force)

    if not success:
        logger.error("Process 3 failed. Stopping execution.")
        exit(1)

    # Process 4: Generate statistics and plots
    if not args.skip_stats:
        success &= process4_generate_statistics(PROTEIN_SEQUENCE_DIR, cfg.dataset, args.force)

        if not success:
            logger.error("Process 4 failed. Dataset preparation completed but statistics generation failed.")
            exit(1)

    logger.info("🎉 Protein sequence dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
