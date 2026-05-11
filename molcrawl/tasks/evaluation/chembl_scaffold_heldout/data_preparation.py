"""Loader for the ChEMBL scaffold held-out split.

The held-out CSV is produced by an optional scaffold-split flag that
will be wired into ``molcrawl.compounds.dataset.prepare_chembl`` as part
of this phase.  Until that wiring is complete, any CSV with at least a
``smiles`` column (and optionally an assay label column) works as input.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_heldout(
    path: Path,
    smiles_column: str = "smiles",
    label_column: Optional[str] = None,
) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if smiles_column not in df.columns:
        raise ValueError(
            f"SMILES column {smiles_column!r} missing from {csv_path}. "
            f"Available: {list(df.columns)}"
        )
    keep: List[str] = [smiles_column]
    if label_column is not None:
        if label_column not in df.columns:
            raise ValueError(
                f"Label column {label_column!r} missing from {csv_path}. "
                f"Available: {list(df.columns)}"
            )
        keep.append(label_column)
    df = df[keep].copy()
    df = df.dropna(subset=[smiles_column]).reset_index(drop=True)
    logger.info("Loaded %d ChEMBL scaffold held-out examples", len(df))
    return df
