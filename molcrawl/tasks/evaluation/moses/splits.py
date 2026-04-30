"""MOSES does not require a custom split.

The reference CSVs ship with their own train / test / test_scaffolds
files.  This module exists only to keep the standard task layout; it
exposes a helper that validates the expected filenames exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


REQUIRED_FILES: Tuple[str, ...] = ("train.csv", "test.csv")


def ensure_reference_files(reference_dir: Path) -> None:
    for name in REQUIRED_FILES:
        if not (Path(reference_dir) / name).exists():
            raise FileNotFoundError(
                f"Expected {name} under {reference_dir}. See README.md for "
                "where to download the MOSES reference split."
            )
