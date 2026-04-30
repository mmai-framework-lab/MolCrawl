#!/usr/bin/env python3
"""
GuacaMol Dataset Download Script

Download the GuacaMol benchmark dataset from Figshare.
https://figshare.com/projects/GuacaMol/56639

How to use:
    python src/preparation/download_guacamol.py

environmental variables:
    LEARNING_SOURCE_DIR: Base directory where dataset is saved (required)
"""

import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# GuacaMol dataset URLs from Figshare
GUACAMOL_URLS = {
    "train": "https://figshare.com/ndownloader/files/13612760",  # guacamol_v1_train.smiles
    "valid": "https://figshare.com/ndownloader/files/13612766",  # guacamol_v1_valid.smiles
    "test": "https://figshare.com/ndownloader/files/13612757",  # guacamol_v1_test.smiles
}


def _download_from_huggingface(output_dir):
    """
    Fallback: download GuacaMol SMILES from HuggingFace ``MolGen/GuacaMol-raw``.

    Used when Figshare downloads are blocked by WAF challenge (HTTP 202,
    content-length 0).  Requires the ``datasets`` package.

    Args:
        output_dir: Directory where ``guacamol_v1_{train,valid,test}.smiles``
            will be written.

    Returns:
        True if all three splits were written successfully, False otherwise.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets package not available — skipping HuggingFace fallback.", file=sys.stderr)
        return False

    output_dir = Path(output_dir)
    # HF split name → Figshare-compatible filename
    SPLIT_MAP = {
        "train": "guacamol_v1_train.smiles",
        "valid": "guacamol_v1_valid.smiles",
        "test": "guacamol_v1_test.smiles",
    }

    print("  ↪ Trying HuggingFace fallback: MolGen/GuacaMol-raw ...")
    try:
        ds = load_dataset("MolGen/GuacaMol-raw", trust_remote_code=True)
    except Exception as e:
        print(f"  ✗ HuggingFace load failed: {e}", file=sys.stderr)
        return False

    success = True
    for hf_split, filename in SPLIT_MAP.items():
        out_path = output_dir / filename
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  ✓ Already exists: {filename}")
            continue
        if hf_split not in ds:
            print(f"  ✗ Split '{hf_split}' not found in HF dataset.", file=sys.stderr)
            success = False
            continue
        smiles_list = ds[hf_split]["SMILES"]
        with open(out_path, "w") as f:
            f.write("\n".join(smiles_list))
        print(f"  ✓ Written {len(smiles_list):,} lines → {filename}")

    return success


def _find_existing_smiles(filename):
    """
    Search sibling learning_source_* directories for an existing non-empty
    copy of *filename*.  Returns the first match or None.
    """
    import glob as _glob

    workspace = Path(__file__).resolve().parents[2]  # repo root
    candidates = sorted(_glob.glob(str(workspace / "learning_source_*")), reverse=True)
    for ls_dir in candidates:
        candidate = Path(ls_dir) / "compounds" / "benchmark" / "GuacaMol" / filename
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def download_file(url, output_path, chunk_size=8192):
    """
    Download and save file from URL.

    Falls back to copying from a sibling learning_source_* directory when the
    remote server blocks automated downloads (e.g. Figshare WAF challenge).

    Args:
        url: Download source URL
        output_path: save the first file
        chunk_size: Chunk size (bytes)
    """
    import shutil as _shutil

    output_path = Path(output_path)

    # Skip if already exists and has content (0-byte files are treated as missing)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"✓ Already exists: {output_path.name}")
        return True
    if output_path.exists() and output_path.stat().st_size == 0:
        print(f"⚠ Found 0-byte file, re-downloading: {output_path.name}")
        output_path.unlink()

    print(f"Downloading {output_path.name}...")

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()

        # get file size
        total_size = int(response.headers.get("content-length", 0))

        # Download with progress bar
        with (
            open(output_path, "wb") as f,
            tqdm(
                desc=output_path.name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Verify we actually got content (WAF blocks return 0 bytes with 200/202)
        if output_path.stat().st_size == 0:
            output_path.unlink()
            raise requests.exceptions.RequestException(
                "Server returned 0 bytes — possible WAF/bot-protection block."
            )

        print(f"✓ Downloaded: {output_path.name}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"✗ HTTP download failed: {e}", file=sys.stderr)
        # delete failed files
        if output_path.exists():
            output_path.unlink()

        # Fallback: copy from an existing learning_source_* directory
        existing = _find_existing_smiles(output_path.name)
        if existing:
            print(f"  ↪ Copying from existing dataset: {existing}", file=sys.stderr)
            _shutil.copy2(existing, output_path)
            print(f"✓ Copied: {output_path.name} ({output_path.stat().st_size:,} bytes)")
            return True

        print(
            f"  No local fallback found for {output_path.name}.\n"
            f"  Download manually from {url} and place in {output_path.parent}/",
            file=sys.stderr,
        )
        return False


def download_guacamol(compounds_dir):
    """
    Download GuacaMol dataset

    Args:
        compounds_dir: compounds directorypath (example: learning_source_XXX/compounds）

    Raises:
        RuntimeError: If download fails
    """
    import logging

    logger = logging.getLogger(__name__)

    output_dir = Path(compounds_dir) / "benchmark" / "GuacaMol"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading GuacaMol benchmark from https://figshare.com/projects/GuacaMol/56639")
    logger.info(f"Destination: {output_dir}")

    success_count = 0
    total_count = len(GUACAMOL_URLS)

    for split, url in GUACAMOL_URLS.items():
        filename = f"guacamol_v1_{split}.smiles"
        output_path = output_dir / filename

        if download_file(url, output_path):
            success_count += 1

    if success_count < total_count:
        logger.warning(
            f"Figshare download incomplete ({success_count}/{total_count}). "
            "Trying HuggingFace fallback..."
        )
        if not _download_from_huggingface(output_dir):
            raise RuntimeError(
                f"GuacaMol download incomplete: {success_count}/{total_count} files downloaded "
                "and HuggingFace fallback also failed."
            )
        logger.info("✓ GuacaMol: All files obtained via HuggingFace fallback.")
    else:
        logger.info(f"✓ GuacaMol: All {total_count} files downloaded successfully")


def main():
    """Download GuacaMol dataset (for standalone execution)"""

    # Check LEARNING_SOURCE_DIR
    learning_source_dir = os.environ.get("LEARNING_SOURCE_DIR")
    if not learning_source_dir:
        print(
            "ERROR: Environment variable 'LEARNING_SOURCE_DIR' is not set.",
            file=sys.stderr,
        )
        print(
            "Please set LEARNING_SOURCE_DIR before running this script:",
            file=sys.stderr,
        )
        print("  export LEARNING_SOURCE_DIR='learning_20251104'", file=sys.stderr)
        sys.exit(1)

    compounds_dir = Path(learning_source_dir) / "compounds"

    try:
        download_guacamol(str(compounds_dir))
        print("\nNext steps:")
        print("  1. Run the GPT-2 preparation script:")
        print(
            f"     LEARNING_SOURCE_DIR={learning_source_dir} python src/compounds/dataset/prepare_gpt2.py assets/configs/compounds.yaml"
        )
        return 0
    except RuntimeError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
