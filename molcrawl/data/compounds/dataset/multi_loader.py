"""
Multi dataset loader

Provides utilities for dynamically loading and combining multiple datasets for use in training.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

from datasets import DatasetDict, concatenate_datasets, load_from_disk

from molcrawl.data.compounds.dataset.dataset_config import (
    DATASET_DEFINITIONS,
    CompoundDatasetType,
    get_dataset_info,
)

logger = logging.getLogger(__name__)


class MultiDatasetLoader:
    """
    Multi dataset loader

    Dynamically load and combine multiple datasets.
    """

    def __init__(self, compounds_dir: Path):
        """
                Args:
        compounds_dir: Path to the compounds directory
        """
        self.compounds_dir = Path(compounds_dir)

    def get_available_datasets(self) -> List[CompoundDatasetType]:
        """
        Get available HuggingFace Dataset format datasets

                Returns:
        List of available dataset types
        """
        available = []
        for dataset_type, info in DATASET_DEFINITIONS.items():
            hf_path = info.get_hf_dataset_path(self.compounds_dir)
            train_path = hf_path / "train"

            if train_path.exists():
                available.append(dataset_type)

        return available

    def load_datasets(
        self,
        dataset_types: Optional[List[Union[str, CompoundDatasetType]]] = None,
        splits: Optional[List[str]] = None,
        combine: bool = True,
    ) -> Union[DatasetDict, dict]:
        """
        Load dataset

                Args:
        dataset_types: List of dataset types to load (all available if None)
        splits: read split(train, valid, test)
        combine: If True, combine all datasets; if False, keep them separate

                Returns:
        If combine=True: DatasetDict {split_name: combined_dataset}
        If combine=False: dict {dataset_type: DatasetDict}
        """
        if splits is None:
            splits = ["train", "valid", "test"]

        # Decide which dataset to load
        enum_types: List[CompoundDatasetType]
        if dataset_types is None:
            enum_types = self.get_available_datasets()
            if not enum_types:
                raise ValueError(
                    f"No HuggingFace datasets found in {self.compounds_dir}/hf_datasets/\n"
                    "Please run the preparation pipeline first."
                )
        else:
            # If it is a string, convert it to Enum
            enum_types = [CompoundDatasetType(dt) if isinstance(dt, str) else dt for dt in dataset_types]  # type: ignore[misc]

        logger.info(f"Loading {len(enum_types)} datasets: {[dt.value for dt in enum_types]}")

        # load each dataset
        loaded_datasets = {}
        for dataset_type in enum_types:
            info = get_dataset_info(dataset_type)
            hf_path = info.get_hf_dataset_path(self.compounds_dir)

            try:
                dataset_dict = {}
                for split in splits:
                    split_path = hf_path / split
                    if split_path.exists():
                        dataset_dict[split] = load_from_disk(str(split_path))
                        logger.info(f"  Loaded {dataset_type.value}/{split}: {len(dataset_dict[split])} samples")
                    else:
                        logger.warning(f"  {dataset_type.value}/{split} not found, skipping")

                if dataset_dict:
                    loaded_datasets[dataset_type] = DatasetDict(dataset_dict)

            except Exception as e:
                logger.error(f"  Failed to load {dataset_type.value}: {e}")
                continue

        if not loaded_datasets:
            raise ValueError("No datasets could be loaded")

        # Combine mode
        if combine:
            return self._combine_datasets(loaded_datasets, splits)
        else:
            return loaded_datasets

    def _combine_datasets(self, loaded_datasets: dict, splits: List[str]) -> DatasetDict:
        """
        Combine multiple datasets

                Args:
        loaded_datasets: Dictionary of {dataset_type: DatasetDict}
        splits: splits to combine

                Returns:
        Combined DatasetDict
        """
        logger.info("Combining datasets...")

        combined = {}
        for split in splits:
            # Collect the same split of each dataset
            split_datasets = []
            for _dataset_type, dataset_dict in loaded_datasets.items():
                if split in dataset_dict:
                    split_datasets.append(dataset_dict[split])

            if split_datasets:
                # join
                combined[split] = concatenate_datasets(split_datasets)
                logger.info(f"  Combined {split}: {len(combined[split])} samples from {len(split_datasets)} datasets")

        return DatasetDict(combined)

    def load_single_dataset(self, dataset_type: Union[str, CompoundDatasetType]) -> DatasetDict:
        """
        Load a single dataset

                Args:
        dataset_type: Dataset type

                Returns:
                    DatasetDict
        """
        if isinstance(dataset_type, str):
            dataset_type = CompoundDatasetType(dataset_type)

        info = get_dataset_info(dataset_type)
        hf_path = info.get_hf_dataset_path(self.compounds_dir)

        if not hf_path.exists():
            raise ValueError(f"Dataset not found: {dataset_type.value} at {hf_path}")

        logger.info(f"Loading {dataset_type.value}...")

        dataset_dict = {}
        for split in ["train", "valid", "test"]:
            split_path = hf_path / split
            if split_path.exists():
                dataset_dict[split] = load_from_disk(str(split_path))
                logger.info(f"  Loaded {split}: {len(dataset_dict[split])} samples")

        return DatasetDict(dataset_dict)


def load_compounds_datasets(
    compounds_dir: Path,
    dataset_types: Optional[List[Union[str, CompoundDatasetType]]] = None,
    splits: Optional[List[str]] = None,
    combine: bool = True,
) -> Union[DatasetDict, dict]:
    """
    Helper function to load compound dataset

        Args:
    compounds_dir: compounds directorypath of
    dataset_types: List of dataset types to load (all available if None)
    splits: read splits
    combine: whether to combine datasets

        Returns:
    Loaded dataset
    """
    if splits is None:
        splits = ["train", "valid", "test"]

    loader = MultiDatasetLoader(compounds_dir)
    return loader.load_datasets(dataset_types, splits, combine)
