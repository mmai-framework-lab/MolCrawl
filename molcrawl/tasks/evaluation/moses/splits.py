"""MOSES does not require a custom split.

The reference CSVs ship with their own train / test / test_scaffolds
files. This module validates the expected filenames exist and gathers
all three reference pools needed by the evaluator (train list +
canonical sets for train / test / test_scaffolds).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

from .data_preparation import canonicalise_set, load_reference_smiles

logger = logging.getLogger(__name__)

REQUIRED_FILES: Tuple[str, ...] = ("train.csv", "test.csv")
OPTIONAL_FILES: Tuple[str, ...] = ("test_scaffolds.csv",)


def ensure_reference_files(reference_dir: Path) -> None:
    for name in REQUIRED_FILES:
        if not (Path(reference_dir) / name).exists():
            raise FileNotFoundError(
                f"Expected {name} under {reference_dir}. See README.md for "
                "where to download the MOSES reference split."
            )


@dataclass
class ReferencePools:
    """Bundle of all reference SMILES pools the evaluator may consult."""

    train: List[str]
    train_canonical: Set[str]
    test_canonical: Set[str]
    scaffolds_canonical: Set[str] = field(default_factory=set)


def prepare_reference_pools(
    reference_dir: Path,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    scaffolds_limit: Optional[int] = None,
    include_scaffolds: bool = True,
) -> ReferencePools:
    """Load the three reference splits and pre-canonicalise their SMILES."""
    ensure_reference_files(reference_dir)
    reference_dir = Path(reference_dir)

    train = load_reference_smiles(reference_dir / "train.csv", limit=train_limit)
    test = load_reference_smiles(reference_dir / "test.csv", limit=test_limit)
    scaffolds: List[str] = []
    if include_scaffolds and (reference_dir / "test_scaffolds.csv").exists():
        scaffolds = load_reference_smiles(
            reference_dir / "test_scaffolds.csv", limit=scaffolds_limit
        )

    logger.info(
        "Canonicalising reference pools (train=%d, test=%d, scaffolds=%d)",
        len(train),
        len(test),
        len(scaffolds),
    )
    return ReferencePools(
        train=train,
        train_canonical=canonicalise_set(train),
        test_canonical=canonicalise_set(test),
        scaffolds_canonical=canonicalise_set(scaffolds) if scaffolds else set(),
    )
