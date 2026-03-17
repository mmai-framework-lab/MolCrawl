import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

# Add project root src directory to path
from molcrawl.config.paths import (
    CLINVAR_DIR,
    CLINVAR_SOURCE_FILE,
    GENOME_SEQUENCE_DIR,
    get_refseq_tokenizer_path,
)
from molcrawl.core.base import setup_logging
from molcrawl.genome_sequence.dataset.refseq.download_refseq import download_refseq
from molcrawl.genome_sequence.dataset.refseq.fasta_to_raw import fasta_to_raw_genome
from molcrawl.genome_sequence.dataset.sentence_piece_tokenizer import train_tokenizer
from molcrawl.genome_sequence.dataset.tokenizer import raw_to_parquet
from molcrawl.genome_sequence.utils.config import GenomeSequenceConfig

logger = logging.getLogger(__name__)


def create_distribution_plot(data):
    """Create and save distribution plot for tokenized sequence lengths"""
    try:
        from molcrawl.utils.image_manager import get_image_path

        plt.hist(data["train"]["num_tokens"], bins=np.arange(0, 200, 1))
        plt.xlabel("Length of tokenized dataset")
        plt.title("Distribution of tokenized lengths")

        image_path = get_image_path("genome_sequence", "genome_sequence_tokenized_lengths_dist.png")
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
        base_dir (str): Base directory for genome sequence data

    Returns:
        bool: True if all steps are completed, False otherwise
    """
    # Marker file path for each processing stage
    download_marker = Path(base_dir) / "download_complete.marker"
    fasta_to_raw_marker = Path(base_dir) / "fasta_to_raw_complete.marker"
    train_tokenizer_marker = Path(base_dir) / "train_tokenizer_complete.marker"
    raw_to_parquet_marker = Path(base_dir) / "raw_to_parquet_complete.marker"

    # Output directory and path to check file existence
    raw_files_dir = Path(base_dir) / "raw_files"
    tokenizer_model = Path(base_dir) / "spm_tokenizer.model"
    parquet_dir = Path(base_dir) / "parquet_files"

    # Check progress
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


def process1_download_refseq(
    base_dir,
    path_species,
    num_worker,
    species_timeout=30 * 60,
    max_retries=2,
    force=False,
):
    """Process 1: Download RefSeq dataset
    Args:
        base_dir (str): Base directory for genome sequence data
        path_species (str): Path to species list file
        num_worker (int): Number of workers for parallel processing
        species_timeout (int): Per-species download timeout in seconds
        max_retries (int): Maximum retries per species before giving up
        force (bool): Force re-download even if already completed
    Returns:
        bool: True if successful, False otherwise
    """
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

        logger.info(f" - Species timeout : {species_timeout}s")
        logger.info(f" - Max retries     : {max_retries}")

        download_refseq(
            base_dir,
            path_species,
            num_worker,
            species_timeout=species_timeout,
            max_retries=max_retries,
        )

        # Check if all species failed — treat that as a hard failure
        failed_path = Path(base_dir) / "failed_species.json"
        if failed_path.exists():
            import json

            with open(failed_path) as fp:
                failed = json.load(fp)
            if len(failed) > 0:
                logger.warning(f"{len(failed)} species failed to download.")
                # Count total species from directory of .txt files
                species_path = Path(path_species)
                if species_path.is_dir():
                    species_count = sum(
                        sum(1 for _ in open(f)) for f in species_path.glob("*.txt") if f.is_file()
                    )
                elif species_path.is_file():
                    species_count = sum(1 for _ in open(species_path))
                else:
                    species_count = 0
                if species_count > 0 and len(failed) >= species_count:
                    logger.error("ALL species failed to download. Not marking step as complete.")
                    return False
                elif len(failed) > 0:
                    logger.warning(f"Continuing despite {len(failed)} failed species.")

        download_marker.touch()
        logger.info("RefSeq download completed.")
        return True

    except Exception as e:
        logger.error(f"RefSeq download failed: {e}")
        return False


def process2_fasta_to_raw(base_dir, num_worker, max_lines_per_file, force=False):
    """Process 2: Convert FASTA files to raw text format
    Args:
        base_dir (str): Base directory for genome sequence data
        num_worker (int): Number of workers for parallel processing
        max_lines_per_file (int): Maximum lines per output file
        force (bool): Force reconversion even if already completed
    Returns:
        bool: True if successful, False otherwise
    """
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

        # Verify output was actually produced
        raw_count = len(list(raw_files_dir.glob("*.raw"))) if raw_files_dir.exists() else 0
        if raw_count == 0:
            logger.error("FASTA to raw conversion produced 0 files. Not marking step as complete.")
            return False

        fasta_to_raw_marker.touch()
        logger.info(f"FASTA to raw conversion completed ({raw_count} files).")
        return True

    except Exception as e:
        logger.error(f"FASTA to raw conversion failed: {e}")
        return False


def process3_train_tokenizer(base_dir, vocab_size, max_lines_per_file, input_sentence_size, force=False):
    """Process 3: Train SentencePiece tokenizer
    Args:
        base_dir (str): Base directory for genome sequence data
        vocab_size (int): Vocabulary size for tokenizer
        max_lines_per_file (int): Maximum lines per file for training
        input_sentence_size (int): Input sentence size for tokenizer
        force (bool): Force retraining even if already completed
    Returns:
        bool: True if successful, False otherwise
    """
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

        # Assuming that the raw_to_parquet side is an implementation that receives num_proc / batch_size
        raw_to_parquet(base_dir, num_proc=num_proc, batch_size=batch_size)

        raw_to_parquet_marker.touch()
        logger.info("Raw to Parquet conversion completed.")
        return True

    except Exception as e:
        logger.error(f"Raw to Parquet conversion failed: {e}")
        return False


def process5_generate_statistics(base_dir, vocab_size, force=False):
    """Process 5: Generate statistics and distribution plots
    Args:
        base_dir (str): Base directory for genome sequence data
        vocab_size (int): Vocabulary size
        force (bool): Force regeneration even if already exists
    Returns:
        bool: True if successful, False otherwise
    """
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

        from molcrawl.utils.image_manager import get_image_path

        plot_file = Path(get_image_path("genome_sequence", "genome_sequence_tokenized_lengths_dist.png"))
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


def process_clinvar_finetune(force: bool = False) -> bool:
    """Prepare the ClinVar language-model fine-tuning dataset.

    If ``$LEARNING_SOURCE_DIR/genome_sequence/clinvar/clinvar_sequences.csv``
    does not exist, it is generated automatically by downloading the
    ``gonzalobenegas/clinvar`` dataset from HuggingFace and embedding each
    variant in its GRCh38 genomic context (requires
    ``dataset/GCA_000001405.28_GRCh38.p13_genomic.fna`` in the project root).

    Tokenises the reference + variant sequences with the genome SentencePiece
    BPE tokenizer and saves a chunked HuggingFace DatasetDict to
    ``$LEARNING_SOURCE_DIR/genome_sequence/clinvar/training_ready_hf_dataset/``.

    Args:
        force: Re-prepare even if outputs already exist.

    Returns:
        True on success, False on failure.
    """
    from molcrawl.genome_sequence.dataset.clinvar.prepare_clinvar import prepare_clinvar

    output_dir = Path(CLINVAR_DIR)
    ready_marker = output_dir / "clinvar_prepare_complete.marker"

    if not force and ready_marker.exists():
        logger.info("👉ClinVar: dataset already prepared. Skipping (use --force to redo).")
        return True

    logger.info("👉ClinVar: Preparing training_ready_hf_dataset from clinvar_sequences.csv...")
    try:
        prepare_clinvar(
            source_file=CLINVAR_SOURCE_FILE,
            output_dir=output_dir,
            tokenizer_path=get_refseq_tokenizer_path(),
        )
        ready_marker.touch()
        logger.info("🎉 ClinVar dataset preparation completed successfully!")
        return True
    except Exception as exc:
        logger.error("ClinVar preparation failed: %s", exc)
        return False


def main():
    """Main function to orchestrate the genome sequence dataset preparation"""
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["refseq", "clinvar"],
        default=["refseq"],
        help="Which dataset(s) to prepare (default: refseq).",
    )
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

    # ── ClinVar fine-tuning branch ─────────────────────────────────────────────
    if "clinvar" in args.datasets:
        if not process_clinvar_finetune(force=args.force):
            exit(1)
        if "refseq" not in args.datasets:
            return

    # ── RefSeq pretraining branch (original pipeline) ─────────────────────────
    # base_dir for heavy processing (specify in config if you want to release to local SSD etc.)
    base_dir = GENOME_SEQUENCE_DIR + getattr(cfg, "local_base_dir", "scratch")
    logger.info(f"Using base_dir: {base_dir}")

    # Check progress
    all_completed = check_progress_status(base_dir)

    if all_completed and not args.force:
        logger.info("All processing steps are already completed!")
        logger.info("Use --force option if you want to reprocess everything.")
        if not args.skip_stats:
            process5_generate_statistics(base_dir, cfg.vocab_size, args.force)
        return

    success = True

    # Process 1: Download RefSeq dataset
    success &= process1_download_refseq(
        base_dir,
        cfg.path_species,
        cfg.num_worker,
        species_timeout=getattr(cfg, "species_timeout", 30 * 60),
        max_retries=getattr(cfg, "max_retries", 2),
        force=args.force,
    )
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

    # Process 4: Convert raw to Parquet (parallel & batch settings)
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
