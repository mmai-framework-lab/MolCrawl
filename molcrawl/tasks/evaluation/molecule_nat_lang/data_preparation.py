"""Loader for molecule-caption pair evaluations."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_pairs(
    path: Path,
    smiles_column: str = "smiles",
    caption_column: str = "caption",
) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    df = pd.read_csv(file_path)
    required = {smiles_column, caption_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"molecule_nat_lang file missing columns {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )
    df = df.dropna(subset=list(required)).reset_index(drop=True)
    logger.info("Loaded %d molecule-caption pairs", len(df))
    return df
