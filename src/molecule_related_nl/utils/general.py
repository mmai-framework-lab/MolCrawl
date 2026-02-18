from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def load_jsonl_dataset(dataset_path: Union[str, Path]):
    """
    Load SMolInstruct dataset from JSONL files in raw/ directory

    Args:
        dataset_path: Path to SMolInstruct directory (contains raw/ subdirectory)

    Returns:
        DatasetDict with train/dev/test splits
    """
    dataset_path_obj = Path(dataset_path)
    raw_dir = dataset_path_obj / "raw"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    splits = {}
    for split_name in ["train", "dev", "test"]:
        split_dir = raw_dir / split_name
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue

        logger.info(f"Loading {split_name} split from {split_dir}")

        # Load all JSONL files in the split directory
        all_data = []
        jsonl_files = list(split_dir.glob("*.jsonl"))
        logger.info(f"Found {len(jsonl_files)} JSONL files in {split_name}")

        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        all_data.append(data)

        logger.info(f"Loaded {len(all_data)} samples for {split_name}")

        # Create Dataset from list of dicts
        # Explicitly define features to ensure proper serialization
        if all_data:
            # Infer features from first sample, all fields as string
            from datasets import Dataset, Features, Value

            features = Features({key: Value("string") for key in all_data[0].keys()})
            splits[split_name] = Dataset.from_list(all_data, features=features)
        else:
            from datasets import Dataset

            splits[split_name] = Dataset.from_list(all_data)

    # Rename 'dev' to 'valid' for consistency
    if "dev" in splits:
        splits["valid"] = splits.pop("dev")

    from datasets import DatasetDict

    return DatasetDict(splits)


def read_dataset(dataset_path: Union[str, Path]):
    """
    Read dataset from disk, supporting JSONL, split directories, and DatasetDict format

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        dict or DatasetDict: Dictionary of splits with Dataset objects
    """
    dataset_path_obj = Path(dataset_path)

    if not dataset_path_obj.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path_obj}")

    ## TODO: duplication check.
    # Check if this is a parquet file
    if dataset_path_obj.is_file() and dataset_path_obj.suffix == ".parquet":
        logger.info(f"Loading parquet file: {dataset_path_obj}")
        from datasets import Dataset
        dataset = Dataset.from_parquet(str(dataset_path_obj))
        
        # If the dataset has a 'split' column, split it accordingly
        if 'split' in dataset.column_names:
            logger.info("Found 'split' column, splitting dataset by split values")
            split_values = dataset.unique('split')
            splits = {}
            for split_name in split_values:
                split_dataset = dataset.filter(lambda x: x['split'] == split_name)
                # Remove the split column as it's no longer needed
                split_dataset = split_dataset.remove_columns(['split'])
                splits[split_name] = split_dataset
                logger.info(f"Created {split_name} split with {len(split_dataset)} samples")
            from datasets import DatasetDict

            return DatasetDict(splits)
        else:
            # Return as DatasetDict with train split
            from datasets import DatasetDict

            return DatasetDict({"train": dataset})

    # Check if this is a SMolInstruct-style directory with raw/ subdirectory
    raw_dir = dataset_path_obj / "raw"
    if raw_dir.exists() and raw_dir.is_dir():
        logger.info(f"Detected JSONL format dataset at {dataset_path_obj}")
        return load_jsonl_dataset(dataset_path_obj)

    # Try to load as DatasetDict first (if it was saved with save_to_disk)
    try:
        logger.info(f"Attempting to load dataset as DatasetDict from {dataset_path_obj}")
        from datasets import DatasetDict

        dataset_dict = DatasetDict.load_from_disk(str(dataset_path_obj))
        logger.info(f"Successfully loaded DatasetDict with splits: {list(dataset_dict.keys())}")
        return dataset_dict
    except Exception as e:
        logger.debug(f"Not a DatasetDict format: {e}")

    # Fall back to loading individual split directories
    splits = {}
    try:
        for folder in os.listdir(dataset_path_obj):
            folder_path = dataset_path_obj / folder
            if folder_path.is_dir():
                # Skip cache and metadata directories
                if folder.startswith(".") or folder == "hf_cache":
                    continue
                try:
                    logger.info(f"Loading split: {folder}")
                    from datasets import Dataset

                    splits[folder] = Dataset.load_from_disk(str(folder_path))
                    logger.info(f"Loaded {folder} with {len(splits[folder])} samples")
                except Exception as split_error:
                    logger.warning(f"Failed to load split {folder}: {split_error}")

        if not splits:
            raise ValueError(f"No valid dataset splits found in {dataset_path_obj}")

        return splits
    except Exception as e:
        logger.error(f"Failed to read dataset from {dataset_path_obj}: {e}")
        raise


def save_dataset(dataset, dataset_path: Union[str, Path]):
    """
    Save dataset to disk or as parquet file

    Args:
        dataset: Dictionary of Dataset objects or DatasetDict
        dataset_path: Path to save the dataset (directory or .parquet file)
    """
    dataset_path_obj = Path(dataset_path)

    # Check if saving as parquet file
    if dataset_path_obj.suffix == ".parquet":
        logger.info(f"Saving dataset as parquet to {dataset_path_obj}")
        os.makedirs(dataset_path_obj.parent, exist_ok=True)

        # Convert to DatasetDict if it's a dict
        from datasets import DatasetDict

        if not isinstance(dataset, DatasetDict):
            dataset = DatasetDict(dataset)

        # Concatenate all splits and save as single parquet
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Combine all splits
        all_data = []
        for split_name, split_dataset in dataset.items():
            # Add split column
            split_data = split_dataset.to_pandas()
            split_data["split"] = split_name
            all_data.append(split_data)

        import pandas as pd

        combined_df = pd.concat(all_data, ignore_index=True)
        table = pa.Table.from_pandas(combined_df)
        pq.write_table(table, dataset_path_obj)
        logger.info(f"Saved {len(combined_df)} samples to parquet file")
        return

    # Otherwise save as directory structure
    os.makedirs(dataset_path_obj, exist_ok=True)
    logger.info(f"Saving dataset to {dataset_path_obj}")

    # If it's a DatasetDict, we can save it directly
    from datasets import DatasetDict

    if isinstance(dataset, DatasetDict):
        dataset.save_to_disk(str(dataset_path_obj))
        logger.info(f"Saved DatasetDict with {len(dataset)} splits")
        return

    # Otherwise, save each split separately
    for split in dataset.keys():
        split_path = dataset_path_obj / split
        logger.info(f"Saving {split} split to {split_path}")
        dataset[split].save_to_disk(str(split_path))
        logger.info(f"Saved {split} with {len(dataset[split])} samples")
