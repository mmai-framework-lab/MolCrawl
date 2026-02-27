#!/usr/bin/env python3
"""
Compare arrow dataset structures between old and new implementations
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_dataset_info(dataset_path: Path, split_name: str):
    """Load dataset and extract structure information"""
    split_path = dataset_path / split_name

    if not split_path.exists():
        # Try with .arrow suffix
        arrow_path = dataset_path / f"{split_name}.arrow"
        if arrow_path.exists():
            split_path = arrow_path
        else:
            # Try alternative names
            alternatives = {"valid": "validation", "validation": "valid"}
            if split_name in alternatives:
                alt_split_path = dataset_path / alternatives[split_name]
                alt_arrow_path = dataset_path / f"{alternatives[split_name]}.arrow"
                if alt_split_path.exists():
                    split_path = alt_split_path
                    logger.info(f"  Using alternative split name: {alternatives[split_name]}")
                elif alt_arrow_path.exists():
                    split_path = alt_arrow_path
                    logger.info(f"  Using alternative split name: {alternatives[split_name]}.arrow")
                else:
                    logger.warning(f"  Split not found: {split_name} or {alternatives[split_name]}")
                    return None
            else:
                logger.warning(f"  Split not found: {split_name}")
                return None

    try:
        dataset = load_from_disk(str(split_path))

        info: dict[str, Any] = {
            "num_samples": len(dataset),
            "num_columns": len(dataset.column_names),
            "column_names": sorted(dataset.column_names),
            "features": {},
            "sample_data": {},
        }

        # Get feature types
        for col_name in dataset.column_names:
            info["features"][col_name] = str(dataset.features[col_name])

        # Get first sample for each column
        if len(dataset) > 0:
            first_sample = dataset[0]
            for col_name in dataset.column_names:
                value = first_sample[col_name]
                if isinstance(value, (list, tuple)):
                    info["sample_data"][col_name] = {
                        "type": "list",
                        "length": len(value),
                        "first_5": value[:5] if len(value) > 0 else [],
                    }
                elif isinstance(value, str):
                    info["sample_data"][col_name] = {
                        "type": "string",
                        "length": len(value),
                        "preview": value[:100] + "..." if len(value) > 100 else value,
                    }
                else:
                    info["sample_data"][col_name] = {"type": type(value).__name__, "value": value}

        return info

    except Exception as e:
        logger.error(f"  Failed to load {split_path}: {e}")
        return None


def compare_structures(old_info, new_info, split_name):
    """Compare two dataset structures and report differences"""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Comparing {split_name} split")
    logger.info(f"{'=' * 70}")

    if old_info is None or new_info is None:
        logger.error("Cannot compare - one or both datasets missing")
        return False

    all_match = True

    # Compare number of samples
    logger.info("\n[Sample Count]")
    logger.info(f"  Old: {old_info['num_samples']:,}")
    logger.info(f"  New: {new_info['num_samples']:,}")
    if old_info["num_samples"] != new_info["num_samples"]:
        logger.warning(f"  ⚠️  Sample count differs by {abs(old_info['num_samples'] - new_info['num_samples']):,}")
        all_match = False
    else:
        logger.info("  ✅ Sample count matches")

    # Compare columns
    logger.info("\n[Columns]")
    logger.info(f"  Old columns ({old_info['num_columns']}): {', '.join(old_info['column_names'])}")
    logger.info(f"  New columns ({new_info['num_columns']}): {', '.join(new_info['column_names'])}")

    old_cols = set(old_info["column_names"])
    new_cols = set(new_info["column_names"])

    missing_in_new = old_cols - new_cols
    added_in_new = new_cols - old_cols
    common_cols = old_cols & new_cols

    if missing_in_new:
        logger.warning(f"  ⚠️  Columns missing in new: {', '.join(sorted(missing_in_new))}")
        all_match = False
    if added_in_new:
        logger.warning(f"  ⚠️  Columns added in new: {', '.join(sorted(added_in_new))}")
        all_match = False
    if not missing_in_new and not added_in_new:
        logger.info("  ✅ All columns match")

    # Compare feature types for common columns
    if common_cols:
        logger.info("\n[Feature Types for Common Columns]")
        type_mismatches = []
        for col in sorted(common_cols):
            old_type = old_info["features"][col]
            new_type = new_info["features"][col]
            if old_type != new_type:
                logger.warning(f"  ⚠️  {col}:")
                logger.warning(f"      Old: {old_type}")
                logger.warning(f"      New: {new_type}")
                type_mismatches.append(col)
                all_match = False

        if not type_mismatches:
            logger.info(f"  ✅ All feature types match for {len(common_cols)} columns")

    # Compare sample data structure
    logger.info("\n[Sample Data Structure]")
    for col in sorted(common_cols):
        old_sample = old_info["sample_data"].get(col, {})
        new_sample = new_info["sample_data"].get(col, {})

        logger.info(f"\n  Column: {col}")
        logger.info(f"    Old: {old_sample.get('type', 'unknown')}")
        logger.info(f"    New: {new_sample.get('type', 'unknown')}")

        if old_sample.get("type") == "list" and new_sample.get("type") == "list":
            logger.info(f"    Old length: {old_sample.get('length', 0)}")
            logger.info(f"    New length: {new_sample.get('length', 0)}")
            if old_sample.get("first_5") and new_sample.get("first_5"):
                logger.info(f"    Old first 5: {old_sample['first_5']}")
                logger.info(f"    New first 5: {new_sample['first_5']}")
        elif old_sample.get("type") == "string" and new_sample.get("type") == "string":
            logger.info(f"    Old preview: {old_sample.get('preview', '')[:80]}")
            logger.info(f"    New preview: {new_sample.get('preview', '')[:80]}")

    return all_match


def main():
    parser = argparse.ArgumentParser(description="Compare arrow dataset structures between old and new implementations")
    parser.add_argument(
        "old_dataset",
        type=str,
        help="Path to old dataset directory (e.g., learning_source/molecule_nl/molecule_related_natural_language_tokenized.parquet)",
    )
    parser.add_argument(
        "new_dataset", type=str, help="Path to new dataset directory (e.g., learning_source/molecule_nl/arrow_splits)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test", "valid"],
        help="Splits to compare (default: train test valid)",
    )

    args = parser.parse_args()

    old_path = Path(args.old_dataset)
    new_path = Path(args.new_dataset)

    if not old_path.exists():
        logger.error(f"Old dataset path does not exist: {old_path}")
        sys.exit(1)

    if not new_path.exists():
        logger.error(f"New dataset path does not exist: {new_path}")
        sys.exit(1)

    logger.info(f"{'=' * 70}")
    logger.info("Dataset Structure Comparison")
    logger.info(f"{'=' * 70}")
    logger.info(f"Old dataset: {old_path}")
    logger.info(f"New dataset: {new_path}")

    all_splits_match = True

    for split_name in args.splits:
        logger.info(f"\n\nLoading {split_name} split from old dataset...")
        old_info = load_dataset_info(old_path, split_name)

        logger.info(f"Loading {split_name} split from new dataset...")
        new_info = load_dataset_info(new_path, split_name)

        if old_info and new_info:
            matches = compare_structures(old_info, new_info, split_name)
            if not matches:
                all_splits_match = False

    # Final summary
    logger.info(f"\n\n{'=' * 70}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'=' * 70}")
    if all_splits_match:
        logger.info("✅ All dataset structures match!")
    else:
        logger.warning("⚠️  Some differences found between old and new datasets")
        logger.info("\nRecommendations:")
        logger.info("1. Review the differences above")
        logger.info("2. Ensure backward compatibility if needed")
        logger.info("3. Update dependent code if structure changed")

    return 0 if all_splits_match else 1


if __name__ == "__main__":
    sys.exit(main())
