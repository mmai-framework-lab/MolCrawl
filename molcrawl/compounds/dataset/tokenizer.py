"""
Individual dataset tokenizer

Provides classes for tokenizing each dataset independently.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from molcrawl.compounds.dataset.dataset_config import DatasetInfo, CompoundDatasetType
from molcrawl.compounds.utils.tokenizer import CompoundsTokenizer, ScaffoldsTokenizer
from molcrawl.core.base import multiprocess_tokenization
from molcrawl.core.utils.image_manager import get_image_path

logger = logging.getLogger(__name__)


class DatasetTokenizer:
    """
    Individual dataset tokenizer

    Tokenize the processed dataset.
    """

    def __init__(
        self,
        dataset_info: DatasetInfo,
        compounds_dir: Path,
        vocab_path: str,
        max_length: int = 256,
        num_processes: int = 2,
    ):
        """
        Args:
            dataset_info: Dataset information
            compounds_dir: Path to the compounds directory
            vocab_path: Vocabulary file path
            max_length: maximum token length
            num_processes: Number of parallel processing processes
        """
        self.dataset_info = dataset_info
        self.compounds_dir = Path(compounds_dir)
        self.vocab_path = vocab_path
        self.max_length = max_length
        self.num_processes = num_processes

        # Initialize tokenizer
        self.mol_tokenizer = CompoundsTokenizer(vocab_path, max_length)
        self.scaffolds_tokenizer = ScaffoldsTokenizer(vocab_path, max_length)

    def tokenize(self, force: bool = False) -> Optional[pa.Table]:
        """
        Tokenize the dataset

        Args:
            force: Force reprocessing flag

        Returns:
            Tokenized table (None in case of error)
        """
        tokenized_path = self.dataset_info.get_tokenized_path(self.compounds_dir)

        # Skip if already tokenized
        if not force and tokenized_path.exists():
            logger.info(f"✓ {self.dataset_info.name}: Already tokenized, skipping")
            return pq.read_table(tokenized_path)

        # load processed data
        processed_path = self.dataset_info.get_processed_path(self.compounds_dir)
        if not processed_path.exists():
            logger.warning(
                f"⚠ {self.dataset_info.name}: Processed data not found at {processed_path}\n  Please run processing first."
            )
            return None

        logger.info(f"🔤 {self.dataset_info.name}: Tokenizing...")

        try:
            # Load data
            table = pq.read_table(processed_path)

            # Tokenize SMILES
            logger.info("  Tokenizing SMILES...")
            table = multiprocess_tokenization(
                self.mol_tokenizer.bulk_tokenizer_parquet,
                table,
                column_name="smiles",
                new_column_name="tokens",
                processes=self.num_processes,
            )

            # Tokenize Scaffolds
            logger.info("  Tokenizing Scaffolds...")
            table = multiprocess_tokenization(
                self.scaffolds_tokenizer.bulk_tokenizer_parquet,
                table,
                column_name="smiles",
                new_column_name="scaffold_tokens",
                processes=self.num_processes,
            )

            # print invalid SMILES statistics
            self._report_invalid_smiles_stats()

            # keep
            tokenized_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, tokenized_path)

            logger.info(f"✓ {self.dataset_info.name}: Tokenized {table.num_rows} samples")
            return table

        except Exception as e:
            logger.error(f"✗ {self.dataset_info.name}: Tokenization failed - {e}")
            return None

    def _report_invalid_smiles_stats(self):
        """Log invalid SMILES statistics"""
        from molcrawl.compounds.utils.preprocessing import get_invalid_smiles_stats

        invalid_count, total_count, invalid_rate, examples = get_invalid_smiles_stats()
        if total_count == 0:
            return

        name = self.dataset_info.name
        logger.info(f"  [{name}] SMILES validation: {invalid_count}/{total_count} invalid ({invalid_rate:.2f}%)")

        if examples:
            logger.info(f"  [{name}] Examples of invalid SMILES:")
            for i, (reason, smiles) in enumerate(examples, 1):
                logger.info(f"    {i}. [{reason}] {smiles}")

        if invalid_rate > 10.0:
            logger.error(
                f"  [{name}] Very high invalid SMILES rate ({invalid_rate:.2f}%). "
                "Data quality issues should be investigated."
            )
        elif invalid_rate > 5.0:
            logger.warning(
                f"  [{name}] High invalid SMILES rate ({invalid_rate:.2f}%). " "This may indicate data quality issues."
            )
        elif invalid_rate > 1.0:
            logger.info(
                f"  [{name}] Moderate invalid SMILES rate ({invalid_rate:.2f}%). "
                "Within acceptable range for large chemical databases."
            )
        else:
            logger.info(f"  [{name}] Low invalid SMILES rate ({invalid_rate:.2f}%). Data quality is good.")


def compute_tokenization_statistics(
    compounds_dir: Path,
    dataset_types: Optional[List[CompoundDatasetType]] = None,
    force: bool = False,
) -> Dict[CompoundDatasetType, dict]:
    """
    Statistical calculation and visualization of tokenized datasets

    Generate a histogram of the token length distribution for each dataset,
    Outputs statistics on the number of samples and number of tokens.

    Args:
        compounds_dir: compounds directorypath of
        dataset_types: List of dataset types to calculate statistics (if None, all tokenized)
        force: force recalculation flag

    Returns:
        Dictionary of {dataset_type: statistics_dict}
    """
    from molcrawl.compounds.dataset.dataset_config import get_dataset_info, DATASET_DEFINITIONS

    # Decide target dataset
    if dataset_types is None:
        dataset_types = []
        for dt, info in DATASET_DEFINITIONS.items():
            tokenized_path = info.get_tokenized_path(compounds_dir)
            if tokenized_path.exists():
                dataset_types.append(dt)

        if not dataset_types:
            logger.warning("No tokenized datasets available for statistics")
            return {}
    else:
        if isinstance(dataset_types[0], str):
            dataset_types = [CompoundDatasetType(dt) for dt in dataset_types]

    logger.info(f"Computing statistics for {len(dataset_types)} datasets: {[dt.value for dt in dataset_types]}")

    all_results = {}
    for dataset_type in dataset_types:
        info = get_dataset_info(dataset_type)
        tokenized_path = info.get_tokenized_path(compounds_dir)

        if not tokenized_path.exists():
            logger.warning(f"⚠ {info.name}: Tokenized data not found, skipping statistics")
            continue

        # Ensure idempotency with marker file
        stats_marker = tokenized_path.parent / f"{info.name}_stats.marker"
        if not force and stats_marker.exists():
            logger.info(f"✓ {info.name}: Statistics already computed, skipping")
            continue

        logger.info(f"📊 {info.name}: Computing statistics...")

        try:
            table = pq.read_table(tokenized_path)
            statistics = {}

            for column_name, display_name in [("tokens", "SMILES"), ("scaffold_tokens", "Scaffolds")]:
                if column_name not in table.column_names:
                    logger.warning(f"  {info.name}: Column '{column_name}' not found, skipping")
                    continue

                series_length = []
                for item in table[column_name]:
                    if item.is_valid:
                        series_length.append(len(item))

                # Generate histogram
                plt.figure()
                plt.hist(series_length, bins=np.arange(0, 200, 1))
                plt.xlabel(f"Length of tokenized {display_name}")
                plt.title(f"[{info.name}] Distribution of tokenized {display_name} lengths")

                image_path = get_image_path(
                    "compounds",
                    f"compounds_{info.name}_tokenized_{display_name}_lengths_dist.png",
                )
                plt.savefig(image_path)
                plt.close()
                logger.info(f"  Saved histogram to {image_path}")

                statistics[f"Number of Samples for {display_name}"] = len(series_length)
                statistics[f"Number of Tokens for {display_name}"] = sum(series_length)

            for key, value in statistics.items():
                logger.info(f"  {info.name}: {key}: {value}")

            stats_marker.touch()
            all_results[dataset_type] = statistics

        except Exception as e:
            logger.error(f"✗ {info.name}: Statistics computation failed - {e}")

    logger.info(f"Successfully computed statistics for {len(all_results)}/{len(dataset_types)} datasets")
    return all_results


def tokenize_all_processed_datasets(
    compounds_dir: Path,
    vocab_path: str,
    max_length: int = 256,
    dataset_types: Optional[List[CompoundDatasetType]] = None,
    force: bool = False,
    num_processes: int = 2,
) -> dict:
    """
    Tokenize all processed datasets

    Args:
        compounds_dir: compounds directorypath of
        vocab_path: Vocabulary file path
        max_length: maximum token length
        dataset_types: List of dataset types to tokenize (if None, all processed)
        force: Force reprocessing flag
        num_processes: Number of parallel processing processes

    Returns:
        Dictionary of {dataset_type: tokenized_table}
    """
    from molcrawl.compounds.dataset.dataset_config import get_dataset_info, DATASET_DEFINITIONS

    # Determine the dataset to be processed
    if dataset_types is None:
        # Get the dataset with processed data
        dataset_types = []
        for dt, info in DATASET_DEFINITIONS.items():
            processed_path = info.get_processed_path(compounds_dir)
            if processed_path.exists():
                dataset_types.append(dt)

        if not dataset_types:
            logger.warning("No processed datasets available for tokenization")
            return {}
    else:
        # If the specified dataset is a string, convert it to an Enum
        if isinstance(dataset_types[0], str):
            dataset_types = [CompoundDatasetType(dt) for dt in dataset_types]

    logger.info(f"Tokenizing {len(dataset_types)} datasets: {[dt.value for dt in dataset_types]}")

    results = {}
    for dataset_type in dataset_types:
        info = get_dataset_info(dataset_type)
        tokenizer = DatasetTokenizer(info, compounds_dir, vocab_path, max_length, num_processes)

        table = tokenizer.tokenize(force=force)
        if table is not None:
            results[dataset_type] = table

    logger.info(f"Successfully tokenized {len(results)}/{len(dataset_types)} datasets")
    return results
