"""ChEBI-20 TSV / CSV loader."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_chebi20(path: Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    suffix = file_path.suffix.lower()
    if suffix in (".tsv", ".txt"):
        df = pd.read_csv(file_path, sep="\t")
    else:
        df = pd.read_csv(file_path)
    required = {"SMILES", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"ChEBI-20 file missing columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )
    df = df.dropna(subset=["SMILES", "description"]).reset_index(drop=True)
    logger.info("Loaded %d ChEBI-20 rows", len(df))
    return df
