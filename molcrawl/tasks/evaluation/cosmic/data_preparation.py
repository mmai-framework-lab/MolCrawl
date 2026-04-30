"""COSMIC table loader with configurable label mapping."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_LABEL_MAP: Mapping[str, int] = {
    "PATHOGENIC": 1,
    "DAMAGING": 1,
    "DRIVER": 1,
    "BENIGN": 0,
    "NEUTRAL": 0,
    "PASSENGER": 0,
}


def load_cosmic(
    path: Path,
    label_column: str = "FATHMM_PREDICTION",
    label_map: Mapping[str, int] = DEFAULT_LABEL_MAP,
) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = {"reference_sequence", "variant_sequence", label_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"COSMIC file missing required columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )
    df["cosmic_label"] = (
        df[label_column].astype(str).str.upper().map(label_map).astype("Int64")
    )
    df = df.dropna(subset=["cosmic_label"]).reset_index(drop=True)
    df["cosmic_label"] = df["cosmic_label"].astype(int)
    logger.info("Loaded %d COSMIC variants", len(df))
    return df
