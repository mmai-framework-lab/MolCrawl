"""DeepLoc 2.0 loader.

Expected columns: ``sequence`` and ``localisation`` (one of 10 canonical
DeepLoc classes).  Optional columns (``kingdom``, ``cluster_id``) are
passed through for diagnostic reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


DEEPLOC_CLASSES: Tuple[str, ...] = (
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
)


def load_deeploc(path: Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = {"sequence", "localisation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DeepLoc file missing columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )
    df = df.dropna(subset=list(required)).reset_index(drop=True)
    logger.info("Loaded %d DeepLoc sequences", len(df))
    return df
