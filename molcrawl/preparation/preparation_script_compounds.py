#!/usr/bin/env python3
"""
Compound dataset preparation script

Partial downloads are supported through individual dataset processing.

Usage example:
    # Process the entire dataset
    python src/preparation/preparation_script_compounds.py assets/configs/compounds.yaml

    # Download only specific datasets
    python src/preparation/preparation_script_compounds.py assets/configs/compounds.yaml \
        --download-only --datasets zinc20 opv

    # Processing and tokenization only (if downloaded)
    python src/preparation/preparation_script_compounds.py assets/configs/compounds.yaml \
        --skip-download

    # forced reprocessing
    python src/preparation/preparation_script_compounds.py assets/configs/compounds.yaml --force
"""

import logging
import logging.config
import os
from argparse import ArgumentParser
from pathlib import Path

from molcrawl.compounds.dataset.dataset_config import (
    CompoundDatasetType,
    get_all_dataset_types,
)
from molcrawl.compounds.dataset.download_chembl import download_chembl
from molcrawl.compounds.dataset.hf_converter import convert_all_tokenized_datasets
from molcrawl.compounds.dataset.prepare_chembl import prepare_chembl
from molcrawl.compounds.dataset.processor import process_all_available_datasets
from molcrawl.compounds.dataset.tokenizer import (
    compute_tokenization_statistics,
    tokenize_all_processed_datasets,
)
from molcrawl.compounds.utils.config import CompoundConfig
from molcrawl.compounds.utils.general import (
    download_llamol_datasets,
    download_opv,
    download_zinc20,
)
from molcrawl.config.paths import CHEMBL_DIR, CHEMBL_SOURCE_DIR, COMPOUNDS_DIR
from molcrawl.core.base import setup_logging
from molcrawl.preparation.download_guacamol import download_guacamol

logger = logging.getLogger(__name__)


def _assert_all_datasets_succeeded(step_name: str, target_datasets: list, succeeded_results: dict):
    """Fail fast when one or more target datasets fail at a pipeline step."""
    succeeded = {dt.value for dt in succeeded_results.keys()}
    expected = {dt.value for dt in target_datasets}
    missing = sorted(expected - succeeded)
    if missing:
        raise RuntimeError(
            f"{step_name} failed for datasets: {', '.join(missing)}. Aborting preprocessing. Please check the error logs above."
        )


def download_datasets_individually(cfg, compounds_dir, dataset_types, force=False):
    """
    Download datasets individually

    Args:
        cfg: configuration object
        compounds_dir: compounds directory
        dataset_types: List of dataset types to download
        force: force redownload flag
    """
    data_dir = os.path.join(compounds_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    for dataset_type in dataset_types:
        marker_file = Path(data_dir) / f"{dataset_type.value}_download.marker"

        if not force and marker_file.exists():
            logger.info(f"✓ {dataset_type.value}: Already downloaded, skipping")
            continue

        logger.info(f"📥 Downloading {dataset_type.value}...")

        try:
            if dataset_type == CompoundDatasetType.ZINC20:
                download_zinc20(compounds_dir)
            elif dataset_type == CompoundDatasetType.OPV:
                download_opv(compounds_dir)
            elif dataset_type in [
                CompoundDatasetType.PC9_GAP,
                CompoundDatasetType.ZINC_QM9,
                CompoundDatasetType.REDDB,
                CompoundDatasetType.CHEMBL,
                CompoundDatasetType.PUBCHEMQC_2017,
                CompoundDatasetType.PUBCHEMQC_2020,
            ]:
                # Download LlaMol dataset all at once
                llamol_marker = Path(data_dir) / "llamol_download.marker"
                if not force and llamol_marker.exists():
                    logger.info("✓ LlaMol datasets: Already downloaded, skipping")
                    continue
                download_llamol_datasets(compounds_dir)
                llamol_marker.touch()
                # Also create individual markers
                for dt in [
                    CompoundDatasetType.PC9_GAP,
                    CompoundDatasetType.ZINC_QM9,
                    CompoundDatasetType.REDDB,
                    CompoundDatasetType.CHEMBL,
                    CompoundDatasetType.PUBCHEMQC_2017,
                    CompoundDatasetType.PUBCHEMQC_2020,
                ]:
                    (Path(data_dir) / f"{dt.value}_download.marker").touch()
                break  # Download LlaMol only once
            elif dataset_type == CompoundDatasetType.GUACAMOL:
                download_guacamol(compounds_dir)
            else:
                logger.warning(f"⚠ Unknown dataset type: {dataset_type.value}")
                continue

            marker_file.touch()
            logger.info(f"✓ {dataset_type.value}: Download completed")

        except Exception as e:
            logger.error(f"✗ {dataset_type.value}: Download failed - {e}")


def process_chembl_finetune(force: bool = False) -> bool:
    """Download ChEMBL 36 from EBI and prepare fine-tuning dataset.

    Runs download_chembl (EBI FTP → SQLite → smiles.txt) followed by
    prepare_chembl (tokenise → 80/10/10 split → HF Dataset).

    Args:
        force: Re-run all steps even if marker files already exist.

    Returns:
        True if both steps succeeded, False otherwise.
    """
    logger.info("=" * 70)
    logger.info("ChEMBL fine-tuning dataset preparation")
    logger.info("=" * 70)

    logger.info("[1/2] Downloading ChEMBL 36 …")
    ok = download_chembl(CHEMBL_SOURCE_DIR, force=force)
    if not ok:
        logger.error("ChEMBL download failed.")
        return False

    logger.info("[2/2] Tokenising and preparing HF Dataset …")
    ok = prepare_chembl(source_dir=CHEMBL_SOURCE_DIR, output_dir=CHEMBL_DIR, force=force)
    if not ok:
        logger.error("ChEMBL preparation failed.")
        return False

    logger.info("ChEMBL fine-tuning dataset ready.")
    return True


def main():
    """Main execution function"""
    _extra_choices = ["chembl_finetune"]
    parser = ArgumentParser(description="Compound dataset preparation script")
    parser.add_argument("config", help="Configuration file path")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[dt.value for dt in get_all_dataset_types()] + _extra_choices,
        help="Dataset to process (all available if not specified). "
        "Use 'chembl_finetune' for the EBI ChEMBL 36 fine-tuning pipeline.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing (overwrite existing files)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Run only download",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip processing (property calculation)",
    )
    parser.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="Skip tokenization",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip conversion to HuggingFace format",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip statistical calculation/visualization",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of parallel processing processes (for physical property calculations, default: 16)",
    )
    parser.add_argument(
        "--tokenization-processes",
        type=int,
        default=2,
        help="Number of parallel processing for tokenization (default: 2)",
    )

    args = parser.parse_args()

    # Load configuration
    cfg = CompoundConfig.from_file(args.config).data_preparation
    compounds_dir = COMPOUNDS_DIR
    os.makedirs(compounds_dir, exist_ok=True)

    # Set up logging
    setup_logging(compounds_dir + "/compounds_logs")

    logger.info("=" * 70)
    logger.info("Compound dataset preparation script (revised version)")
    logger.info("=" * 70)
    logger.info(f"Compounds directory: {compounds_dir}")

    # ── ChEMBL fine-tuning (separate EBI pipeline) ───────────────────────────
    if args.datasets and "chembl_finetune" in args.datasets:
        success = process_chembl_finetune(force=args.force)
        import sys

        sys.exit(0 if success else 1)

    # Decide which dataset to process
    if args.datasets:
        dataset_types = [CompoundDatasetType(dt) for dt in args.datasets]
        logger.info(f"Target datasets: {[dt.value for dt in dataset_types]}")
    else:
        dataset_types = None
        logger.info("Target datasets: All available")

    # Step 1: Download
    if not args.skip_download and not args.skip_process and not args.skip_tokenize and not args.skip_convert:
        run_download = True
    elif args.download_only:
        run_download = True
    else:
        run_download = not args.skip_download

    if run_download:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: Dataset download")
        logger.info("=" * 70)

        if dataset_types:
            download_datasets_individually(cfg, compounds_dir, dataset_types, args.force)
        else:
            # Download all datasets
            all_types = list(get_all_dataset_types())
            download_datasets_individually(cfg, compounds_dir, all_types, args.force)

    if args.download_only:
        logger.info("\n✅ Only download completed")
        return

    try:
        target_dataset_types = [CompoundDatasetType(dt) for dt in args.datasets] if args.datasets else None

        # Step 2: Processing (physical property calculation)
        if not args.skip_process:
            logger.info("\n" + "=" * 70)
            logger.info("STEP 2: Dataset processing (physical property calculation)")
            logger.info("=" * 70)

            processed = process_all_available_datasets(
                Path(compounds_dir),
                dataset_types=target_dataset_types,
                force=args.force,
                num_processes=args.num_processes,
            )

            logger.info(f"\n✓ {len(processed)} datasets processed")
            if target_dataset_types is not None:
                _assert_all_datasets_succeeded("STEP 2 processing", target_dataset_types, processed)

        # Step 3: Tokenize
        if not args.skip_tokenize:
            logger.info("\n" + "=" * 70)
            logger.info("STEP 3: Tokenize")
            logger.info("=" * 70)

            tokenized = tokenize_all_processed_datasets(
                Path(compounds_dir),
                cfg.vocab_path,
                cfg.max_length,
                dataset_types=target_dataset_types,
                force=args.force,
                num_processes=args.tokenization_processes,
            )

            logger.info(f"\n✓ {len(tokenized)} datasets tokenized")
            if target_dataset_types is not None:
                _assert_all_datasets_succeeded("STEP 3 tokenization", target_dataset_types, tokenized)

        # Step 4: Convert to HuggingFace format
        if not args.skip_convert:
            logger.info("\n" + "=" * 70)
            logger.info("STEP 4: Convert to HuggingFace Dataset format")
            logger.info("=" * 70)

            converted = convert_all_tokenized_datasets(
                Path(compounds_dir),
                dataset_types=target_dataset_types,
                train_ratio=0.9,
                valid_ratio=0.05,
                test_ratio=0.05,
                force=args.force,
            )

            logger.info(f"\n✓ {len(converted)} datasets converted")
            if target_dataset_types is not None:
                _assert_all_datasets_succeeded("STEP 4 conversion", target_dataset_types, converted)

        # Step 5: Statistical calculation/visualization
        if not args.skip_stats:
            logger.info("\n" + "=" * 70)
            logger.info("STEP 5: Statistical calculation/visualization")
            logger.info("=" * 70)

            stats = compute_tokenization_statistics(
                Path(compounds_dir),
                dataset_types=target_dataset_types,
                force=args.force,
            )

            logger.info(f"\n✓ {len(stats)} datasets statistics computed")
            if target_dataset_types is not None:
                _assert_all_datasets_succeeded("STEP 5 statistics", target_dataset_types, stats)
    except RuntimeError as e:
        logger.error("=" * 70)
        logger.error(f"❌ Preprocessing aborted: {e}")
        logger.error("=" * 70)
        raise SystemExit(1) from e

    logger.info("\n" + "=" * 70)
    logger.info("✅ All processing completed")
    logger.info("=" * 70)
    logger.info(f"Output directory: {compounds_dir}")
    logger.info(f"  - Processed data: {compounds_dir}/processed/")
    logger.info(f"  - Tokenized data: {compounds_dir}/tokenized/")
    logger.info(f"  - HuggingFace datasets: {compounds_dir}/hf_datasets/")


if __name__ == "__main__":
    main()
