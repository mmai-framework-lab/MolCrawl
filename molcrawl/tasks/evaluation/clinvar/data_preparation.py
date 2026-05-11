"""ClinVar data loading utilities.

Keeps the data-shape contract the old ``evaluation.gpt2.clinvar_evaluation``
relied on (a dataframe with ``reference_sequence``, ``variant_sequence``,
``ClinicalSignificance``), and provides a canonical binary
``pathogenic`` column.  The actual bulk download / preprocessing lives in
``molcrawl/evaluation/gpt2/clinvar_data_preparation.py`` and will be
moved in a later PR; this module intentionally only handles in-memory
normalisation so the new evaluator is self-contained.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

PATHOGENIC_TERMS: List[str] = [
    "pathogenic",
    "likely pathogenic",
    "pathogenic/likely pathogenic",
]
BENIGN_TERMS: List[str] = ["benign", "likely benign", "benign/likely benign"]


def load_clinvar(path: str) -> pd.DataFrame:
    """Load a ClinVar-derived table from CSV, TSV, or JSON.

    The file must expose ``reference_sequence``, ``variant_sequence``, and
    ``ClinicalSignificance`` columns.  Additional columns are preserved
    untouched for downstream reporting.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix == ".tsv":
        df = pd.read_csv(file_path, sep="\t")
    elif suffix == ".json":
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    required = {"reference_sequence", "variant_sequence", "ClinicalSignificance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"ClinVar file missing required columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )

    df = add_pathogenic_label(df)
    logger.info(
        "Loaded %d ClinVar variants (pathogenic=%d, benign=%d)",
        len(df),
        int((df["pathogenic"] == 1).sum()),
        int((df["pathogenic"] == 0).sum()),
    )
    return df


def add_pathogenic_label(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a binary ``pathogenic`` column, dropping unknown entries."""

    def classify(value: object) -> object:
        if pd.isna(value):
            return None
        text = str(value).lower()
        if any(term in text for term in PATHOGENIC_TERMS):
            return 1
        if any(term in text for term in BENIGN_TERMS):
            return 0
        return None

    labelled = df.copy()
    labelled["pathogenic"] = labelled["ClinicalSignificance"].apply(classify)
    labelled = labelled.dropna(subset=["pathogenic"])
    labelled["pathogenic"] = labelled["pathogenic"].astype(int)
    return labelled
