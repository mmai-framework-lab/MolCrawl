#!/usr/bin/env python3
"""
Convert a combined parquet file with split column into separate arrow files
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path

import pyarrow.parquet as pq
from datasets import Dataset

logger = logging.getLogger(__name__)


def convert_parquet_to_arrow(parquet_path: str, output_dir: str):
    """
    Convert a parquet file with split column into separate arrow files

    Args:
        parquet_path: Path to the input parquet file
        output_dir: Directory to save the arrow files
    """
    parquet_path_obj = Path(parquet_path)
    output_dir_obj = Path(output_dir)

    if not parquet_path_obj.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    logger.info(f"Reading parquet file from {parquet_path}")

    # Read the parquet file
    table = pq.read_table(str(parquet_path_obj))
    df = table.to_pandas()

    logger.info(f"Loaded {len(df)} total samples")

    # Check if split column exists
    if "split" not in df.columns:
        raise ValueError("Parquet file does not contain 'split' column")

    # Get unique splits
    splits = df["split"].unique()
    logger.info(f"Found splits: {list(splits)}")

    # Create output directory
    os.makedirs(output_dir_obj, exist_ok=True)

    # Process each split
    for split_name in splits:
        logger.info(f"Processing {split_name} split...")

        # Filter data for this split
        split_df = df[df["split"] == split_name].copy()

        # Remove the split column (no longer needed)
        split_df = split_df.drop(columns=["split"])

        logger.info(f"  {split_name}: {len(split_df)} samples")

        # Convert to Dataset
        dataset = Dataset.from_pandas(split_df)

        # Save as arrow file
        output_path = output_dir_obj / f"{split_name}.arrow"
        logger.info(f"  Saving to {output_path}")

        # Save to disk in arrow format
        dataset.save_to_disk(str(output_path))

        logger.info(f"  Successfully saved {split_name} split")

    logger.info("=" * 70)
    logger.info("Conversion completed successfully!")
    logger.info(f"Output directory: {output_dir_obj}")
    logger.info("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Convert parquet file with split column to separate arrow files")
    parser.add_argument("parquet_file", type=str, help="Path to the input parquet file")
    parser.add_argument("output_dir", type=str, help="Directory to save the arrow files")

    args = parser.parse_args()

    try:
        convert_parquet_to_arrow(args.parquet_file, args.output_dir)
    except Exception as e:
        logger.error(f"Failed to convert: {e}")
        sys.exit(1)
