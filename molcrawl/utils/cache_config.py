"""
Cache configuration for Hugging Face libraries.

This module automatically configures cache directories for Hugging Face Datasets
and Transformers based on the LEARNING_SOURCE_DIR environment variable.
"""

import os
import sys
from pathlib import Path


def setup_cache_env() -> None:
    """
    Automatically configure cache directories based on LEARNING_SOURCE_DIR.

    This function should be called at the beginning of scripts that use
    Hugging Face Datasets or Transformers to ensure cache directories
    are configured correctly.

    The cache directories are automatically set to:
    - HF_DATASETS_CACHE: {LEARNING_SOURCE_DIR}/.cache/huggingface/datasets
    - HF_HOME: {LEARNING_SOURCE_DIR}/.cache/huggingface

    Raises:
        SystemExit: If LEARNING_SOURCE_DIR is not set.

    Example:
        >>> from utils.cache_config import setup_cache_env
        >>> setup_cache_env()
    """
    learning_source_dir = os.environ.get("LEARNING_SOURCE_DIR")

    if not learning_source_dir:
        print(
            "❌ ERROR: LEARNING_SOURCE_DIR environment variable is required!",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        print("Please set it before running:", file=sys.stderr)
        print("  export LEARNING_SOURCE_DIR='dir'", file=sys.stderr)
        print(f"  python {sys.argv[0] if sys.argv else 'script'}", file=sys.stderr)
        sys.exit(1)

    # Automatically set cache directories based on LEARNING_SOURCE_DIR
    cache_base = Path(learning_source_dir) / ".cache" / "huggingface"
    hf_datasets_cache = str(cache_base / "datasets")
    hf_home = str(cache_base)

    # Set environment variables
    os.environ["HF_DATASETS_CACHE"] = hf_datasets_cache
    os.environ["HF_HOME"] = hf_home

    print("✓ Cache directories auto-configured:")
    print(f"  HF_DATASETS_CACHE: {hf_datasets_cache}")
    print(f"  HF_HOME: {hf_home}")


if __name__ == "__main__":
    # Test the cache configuration
    try:
        setup_cache_env()
        print("\n✓ Cache directories auto-configured successfully!")
    except SystemExit:
        print("\n✗ Error: LEARNING_SOURCE_DIR environment variable is not set")
        exit(1)
