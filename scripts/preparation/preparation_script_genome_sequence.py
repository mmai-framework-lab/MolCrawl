from argparse import ArgumentParser
from pathlib import Path
import logging
import os
import sys

from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

# プロジェクトルートのsrcディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from genome_sequence.dataset.refseq.download_refseq import download_refseq
from genome_sequence.dataset.refseq.fasta_to_raw import fasta_to_raw_genome
from genome_sequence.dataset.sentence_piece_tokenizer import train_tokenizer
from genome_sequence.dataset.tokenizer import raw_to_parquet
from genome_sequence.utils.config import GenomeSequenceConfig
from core.base import setup_logging
from config.paths import GENOME_SEQUENCE_DIR

logger = logging.getLogger(__name__)


def create_distribution_plot(data):
    """Create and save distribution plot for tokenized sequence lengths"""
    try:
        plt.hist(data["train"]["num_tokens"], bins=np.arange(0, 200, 1))
        plt.xlabel("Length of tokenized dataset")
        plt.title("Distribution of tokenized lengths")
        plt.savefig("assets/img/genome_sequence_tokenized_lengths_dist.png")
        plt.close()
        logger.info(
            "Saved distribution of tokenized dataset lengths to assets/img/genome_sequence_tokenized_lengths_dist.png"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to create distribution plot: {e}")
        return False


def check_progress_status(base_dir):
    """Check the progress status of all processing steps

    Args:
        base_dir (str): Base directory for genome sequence data

    Returns:
        bool: True if all steps are completed, False otherwise
    """
    # 各処理段階のマーカーファイルパス
    download_marker = Path(base_dir) / "download_complete.marker"
    fasta_to_raw_marker = Path(base_dir) / "fasta_to_raw_complete.marker"
    train_tokenizer_marker = Path(base_dir) / "train_tokenizer_complete.marker"
    raw_to_parquet_marker = Path(base_dir) / "raw_to_parquet_complete.marker"

    # 出力ディレクトリとファイルの存在確認用パス
    raw_files_dir = Path(base_dir) / "raw_files"
    tokenizer_model = Path(base_dir) / "spm_tokenizer.model"
    parquet_dir = Path(base_dir) / "parquet_files"

    # 進捗状況の確認
    logger.info("=== Genome Sequence Dataset Preparation Progress ===")
    steps_completed = 0
    total_steps = 4

    if download_marker.exists():
        logger.info("✓ Step 1/4: RefSeq download - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 1/4: RefSeq download - PENDING")

    if fasta_to_raw_marker.exists() and raw_files_dir.exists() and any(raw_files_dir.glob("*.raw")):
        logger.info("✓ Step 2/4: FASTA to raw conversion - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 2/4: FASTA to raw conversion - PENDING")

    if train_tokenizer_marker.exists() and tokenizer_model.exists():
        logger.info("✓ Step 3/4: Tokenizer training - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 3/4: Tokenizer training - PENDING")

    if raw_to_parquet_marker.exists() and parquet_dir.exists() and any(parquet_dir.glob("*.parquet")):
        logger.info("✓ Step 4/4: Raw to Parquet conversion - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 4/4: Raw to Parquet conversion - PENDING")

    logger.info(f"Progress: {steps_completed}/{total_steps} steps completed")
    logger.info("====================================================")

    return steps_completed == total_steps


def process1_download_refseq(base_dir, path_species, num_worker, force=False):
    """Process 1: Download RefSeq dataset"""
    download_marker = Path(base_dir) / "download_complete.marker"

    if not force and download_marker.exists():
        logger.info("👉Process1 : RefSeq dataset download already completed. Skipping...")
        logger.info("Use --force option to re-download.")
        return True

    try:
        if force:
            logger.info("👉Process1 : Force option specified. Re-downloading RefSeq dataset...")
        else:
            logger.info("👉Process1 : Downloading RefSeq dataset...")

        download_refseq(base_dir, path_species, num_worker)
        download_marker.touch()
        logger.info("RefSeq download completed.")
        return True

    except Exception as e:
        logger.error(f"RefSeq download failed: {e}")
        return False


def process2_fasta_to_raw(base_dir, num_worker, max_lines_per_file, force=False):
    """Process 2: Convert FASTA files to raw text format"""
    fasta_to_raw_marker = Path(base_dir) / "fasta_to_raw_complete.marker"
    raw_files_dir = Path(base_dir) / "raw_files"

    if not force and fasta_to_raw_marker.exists() and raw_files_dir.exists() and any(raw_files_dir.glob("*.raw")):
        logger.info("👉Process2 : FASTA to raw conversion already completed. Skipping...")
        logger.info("Use --force option to reconvert.")
        return True

    try:
        if force:
            logger.info("👉Process2 : Force option specified. Reconverting FASTA to raw text...")
        else:
            logger.info("👉Process2 : Converting FASTA to raw text...")

        logger.info(f" - Base Directory : {base_dir}")
        logger.info(f" - Number of Workers : {num_worker}")
        logger.info(f" - Max Lines per File : {max_lines_per_file}")

        fasta_to_raw_genome(base_dir, num_worker, max_lines_per_file)
        fasta_to_raw_marker.touch()
        logger.info("FASTA to raw conversion completed.")
        return True

    except Exception as e:
        logger.error(f"FASTA to raw conversion failed: {e}")
        return False


def process3_train_tokenizer(base_dir, vocab_size, max_lines_per_file, input_sentence_size, force=False):
    """Process 3: Train SentencePiece tokenizer"""
    train_tokenizer_marker = Path(base_dir) / "train_tokenizer_complete.marker"
    tokenizer_model = Path(base_dir) / "spm_tokenizer.model"

    if not force and train_tokenizer_marker.exists() and tokenizer_model.exists():
        logger.info("👉Process3 : Tokenizer training already completed. Skipping...")
        logger.info("Use --force option to retrain tokenizer.")
        return True

    try:
        if force:
            logger.info("👉Process3 : Force option specified. Retraining tokenizer...")
        else:
            logger.info("👉Process3 : Training tokenizer...")

        logger.info(f" - Base Directory : {base_dir}")
        logger.info(f" - vocab size : {vocab_size}")
        logger.info(f" - max lines per file : {max_lines_per_file}")
        logger.info(f" - input sentence size : {input_sentence_size}")

        train_tokenizer(base_dir, vocab_size, max_lines_per_file, input_sentence_size)
        train_tokenizer_marker.touch()
        logger.info("Tokenizer training completed.")
        return True

    except Exception as e:
        logger.error(f"Tokenizer training failed: {e}")
        return False


def process4_raw_to_parquet(base_dir, num_proc=None, batch_size=None, force=False):
    """Process 4: Convert raw text files to Parquet format"""
    raw_to_parquet_marker = Path(base_dir) / "raw_to_parquet_complete.marker"
    parquet_dir = Path(base_dir) / "parquet_files"

    if not force and raw_to_parquet_marker.exists() and parquet_dir.exists() and any(parquet_dir.glob("*.parquet")):
        logger.info("👉Process4 : Raw to Parquet conversion already completed. Skipping...")
        logger.info("Use --force option to reconvert.")
        return True

    try:
        if force:
            logger.info("👉Process4 : Force option specified. Reconverting raw text to Parquet...")
        else:
            logger.info("👉Process4 : Converting raw text to Parquet...")

        logger.info(f" - Base Directory : {base_dir}")
        if num_proc is not None:
            logger.info(f" - num_proc for map : {num_proc}")
        if batch_size is not None:
            logger.info(f" - batch_size for map : {batch_size}")

        # raw_to_parquet 側が num_proc / batch_size を受け取る実装になっている前提
        raw_to_parquet(base_dir, num_proc=num_proc, batch_size=batch_size)

        raw_to_parquet_marker.touch()
        logger.info("Raw to Parquet conversion completed.")
        return True

    except Exception as e:
        logger.error(f"Raw to Parquet conversion failed: {e}")
        return False


def process5_generate_statistics(base_dir, vocab_size, force=False):
    """Process 5: Generate statistics and distribution plots"""
    logger.info("👉Process5 : Loading Parquet dataset and generating statistics...")

    try:
        data = load_dataset(
            "parquet",
            data_files=[str(Path(base_dir) / "parquet_files")],
            cache_dir=str(Path(base_dir) / "hf_cache"),
        )

        logger.info("👍Dataset loaded successfully.")
        logger.info(f"Number of sequence: {len(data['train'])}")
        logger.info(f"Size of the vocabulary: {vocab_size}")
        logger.info(f"Number of tokens: {sum(data['train']['num_tokens'])}")

        plot_file = Path("assets/img/genome_sequence_tokenized_lengths_dist.png")
        if force or not plot_file.exists():
            if force:
                logger.info("Force option specified. Regenerating distribution plot...")
            logger.info("Creating distribution plot...")

            if not create_distribution_plot(data):
                logger.warning("Distribution plot generation failed, but continuing...")
        else:
            logger.info("Distribution plot already exists. Skipping plot generation.")
            logger.info("Use --force option to regenerate plot.")

        return True

    except Exception as e:
        logger.error(f"Failed to load or process final dataset: {e}")
        return False


def main():
    """Main function to orchestrate the genome sequence dataset preparation"""
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

    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation
    setup_logging(GENOME_SEQUENCE_DIR)

    # 重い処理用 base_dir（ローカルSSDなどに逃がしたい場合は config で指定）
    base_dir = getattr(cfg, "local_base_dir", GENOME_SEQUENCE_DIR)
    logger.info(f"Using base_dir: {base_dir}")

    # 進捗状況の確認
    all_completed = check_progress_status(base_dir)

    if all_completed and not args.force:
        logger.info("All processing steps are already completed!")
        logger.info("Use --force option if you want to reprocess everything.")
        if not args.skip_stats:
            process5_generate_statistics(base_dir, cfg.vocab_size, args.force)
        return

    success = True

    # Process 1: Download RefSeq dataset
    success &= process1_download_refseq(base_dir, cfg.path_species, cfg.num_worker, args.force)
    if not success:
        logger.error("Process 1 failed. Stopping execution.")
        exit(1)

    # Process 2: Convert FASTA to raw text
    success &= process2_fasta_to_raw(base_dir, cfg.num_worker, cfg.max_lines_per_file, args.force)
    if not success:
        logger.error("Process 2 failed. Stopping execution.")
        exit(1)

    # Process 3: Train tokenizer
    success &= process3_train_tokenizer(
        base_dir,
        cfg.vocab_size,
        cfg.max_lines_per_file,
        cfg.input_sentence_size,
        args.force,
    )
    if not success:
        logger.error("Process 3 failed. Stopping execution.")
        exit(1)

    # Process 4: Convert raw to Parquet（並列＆バッチ設定）
    num_proc_parquet = getattr(cfg, "num_proc_parquet", cfg.num_worker)
    batch_size_parquet = getattr(cfg, "parquet_batch_size", 512)

    success &= process4_raw_to_parquet(
        base_dir,
        num_proc=num_proc_parquet,
        batch_size=batch_size_parquet,
        force=args.force,
    )
    if not success:
        logger.error("Process 4 failed. Stopping execution.")
        exit(1)

    # Process 5: Generate statistics and plots
    if not args.skip_stats:
        success &= process5_generate_statistics(base_dir, cfg.vocab_size, args.force)
        if not success:
            logger.error("Process 5 failed. Dataset preparation completed but statistics generation failed.")
            exit(1)

    logger.info("🎉 Genome sequence dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
