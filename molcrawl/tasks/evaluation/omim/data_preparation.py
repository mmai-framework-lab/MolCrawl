"""OMIM loader.

The upstream scripts emit CSVs with ``reference_sequence``,
``variant_sequence``, and ``disease_category`` (``known`` / ``unknown``
or a free-form disease name).  This module keeps the same contract and
exposes a binary ``omim_label`` derived from ``disease_category``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def load_omim(
    path: Path,
    category_column: str = "disease_category",
    positive_terms: Iterable[str] = ("known", "disease", "mendelian"),
) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = {"reference_sequence", "variant_sequence", category_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"OMIM file missing required columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )
    positive_terms = {term.lower() for term in positive_terms}

    def classify(value: object) -> int:
        text = str(value).lower()
        return 1 if any(term in text for term in positive_terms) else 0

    df["omim_label"] = df[category_column].map(classify).astype(int)
    df = df.dropna(subset=["omim_label"]).reset_index(drop=True)
    logger.info("Loaded %d OMIM variants", len(df))
    return df
