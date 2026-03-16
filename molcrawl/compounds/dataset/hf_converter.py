"""
Conversion to HuggingFace Dataset format

Convert tokenized data to HuggingFace Dataset format.
"""

import logging
from pathlib import Path
from typing import Optional, List

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict

from molcrawl.compounds.dataset.dataset_config import DatasetInfo, CompoundDatasetType

logger = logging.getLogger(__name__)


class HFDatasetConverter:
    """
    HuggingFace Dataset transformation class

    Convert tokenized data to HuggingFace Dataset format.
    """

    def __init__(self, dataset_info: DatasetInfo, compounds_dir: Path):
        """
                Args:
        dataset_info: Dataset information
        compounds_dir: Path to the compounds directory
        """
        self.dataset_info = dataset_info
        self.compounds_dir = Path(compounds_dir)

    def convert(
        self,
        train_ratio: float = 0.9,
        valid_ratio: float = 0.05,
        test_ratio: float = 0.05,
        force: bool = False,
        random_seed: int = 42,
    ) -> Optional[DatasetDict]:
        """
        Convert to HuggingFace Dataset format

                Args:
        train_ratio: Ratio of training data
        valid_ratio: Ratio of valid data
        test_ratio: Ratio of test data
        force: forced reconversion flag
        random_seed: random seed

                Returns:
        DatasetDict (None in case of error)
        """
        hf_path = self.dataset_info.get_hf_dataset_path(self.compounds_dir)

        # Skip if already converted
        if not force and hf_path.exists():
            try:
                logger.info(f"✓ {self.dataset_info.name}: Already converted, loading from {hf_path}")
                # Check if train/valid/test all exist
                train_path = hf_path / "train"
                valid_path = hf_path / "valid"
                test_path = hf_path / "test"

                if train_path.exists() and valid_path.exists() and test_path.exists():
                    return DatasetDict(
                        {
                            "train": Dataset.load_from_disk(str(train_path)),
                            "valid": Dataset.load_from_disk(str(valid_path)),
                            "test": Dataset.load_from_disk(str(test_path)),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to load existing dataset: {e}, will reconvert")

        # Load tokenized data
        tokenized_path = self.dataset_info.get_tokenized_path(self.compounds_dir)
        if not tokenized_path.exists():
            logger.warning(
                f"⚠ {self.dataset_info.name}: Tokenized data not found at {tokenized_path}\n  Please run tokenization first."
            )
            return None

        logger.info(f"🔄 {self.dataset_info.name}: Converting to HuggingFace Dataset format...")

        try:
            # Load data
            table = pq.read_table(tokenized_path)
            df = table.to_pandas()

            # data split
            df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            total_samples = len(df)

            train_end = int(total_samples * train_ratio)
            valid_end = train_end + int(total_samples * valid_ratio)

            train_df = df[:train_end]
            valid_df = df[train_end:valid_end]
            test_df = df[valid_end:]

            logger.info(f"  Split: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

            # Convert to HuggingFace Dataset format
            dataset_dict = DatasetDict(
                {
                    "train": Dataset.from_pandas(train_df, preserve_index=False),
                    "valid": Dataset.from_pandas(valid_df, preserve_index=False),
                    "test": Dataset.from_pandas(test_df, preserve_index=False),
                }
            )

            # Save as DatasetDict so `load_from_disk(hf_path)` works directly.
            hf_path.mkdir(parents=True, exist_ok=True)
            dataset_dict.save_to_disk(str(hf_path))
            for split_name, split_dataset in dataset_dict.items():
                logger.info(f"  Saved {split_name}: {len(split_dataset)} samples to {hf_path / split_name}")

            logger.info(f"✓ {self.dataset_info.name}: Converted to HuggingFace Dataset format")
            return dataset_dict

        except Exception as e:
            logger.error(f"✗ {self.dataset_info.name}: Conversion failed - {e}")
            return None


def convert_all_tokenized_datasets(
    compounds_dir: Path,
    dataset_types: Optional[List[CompoundDatasetType]] = None,
    train_ratio: float = 0.9,
    valid_ratio: float = 0.05,
    test_ratio: float = 0.05,
    force: bool = False,
    random_seed: int = 42,
) -> dict:
    """
    Convert all tokenized datasets to HuggingFace format

        Args:
    compounds_dir: compounds directorypath of
    dataset_types: List of dataset types to convert (if None, all tokenized)
    train_ratio: Ratio of training data
    valid_ratio: Ratio of valid data
    test_ratio: Ratio of test data
    force: forced reconversion flag
    random_seed: random seed

        Returns:
    Dictionary of {dataset_type: dataset_dict}
    """
    from molcrawl.compounds.dataset.dataset_config import get_dataset_info, DATASET_DEFINITIONS

    # Determine the dataset to be processed
    if dataset_types is None:
        # Get the dataset with tokenized data
        dataset_types = []
        for dt, info in DATASET_DEFINITIONS.items():
            tokenized_path = info.get_tokenized_path(compounds_dir)
            if tokenized_path.exists():
                dataset_types.append(dt)

        if not dataset_types:
            logger.warning("No tokenized datasets available for conversion")
            return {}
    else:
        # If the specified dataset is a string, convert it to an Enum
        if isinstance(dataset_types[0], str):
            dataset_types = [CompoundDatasetType(dt) for dt in dataset_types]

    logger.info(f"Converting {len(dataset_types)} datasets: {[dt.value for dt in dataset_types]}")

    results = {}
    for dataset_type in dataset_types:
        info = get_dataset_info(dataset_type)
        converter = HFDatasetConverter(info, compounds_dir)

        dataset_dict = converter.convert(
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            force=force,
            random_seed=random_seed,
        )

        if dataset_dict is not None:
            results[dataset_type] = dataset_dict

    logger.info(f"Successfully converted {len(results)}/{len(dataset_types)} datasets")
    return results
