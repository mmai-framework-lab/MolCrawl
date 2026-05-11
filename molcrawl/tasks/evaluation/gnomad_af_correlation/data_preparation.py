"""gnomAD table loader.

Expects a pre-processed table with columns ``reference_sequence``,
``variant_sequence``, and ``allele_frequency``.  The heavy VCF
parsing / gnomAD downloads are handled outside this module by
``workflows/eval-gnomad.sh``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_gnomad(path: Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = {"reference_sequence", "variant_sequence", "allele_frequency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"gnomAD file missing required columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )
    df = df.dropna(subset=list(required)).reset_index(drop=True)
    df["allele_frequency"] = df["allele_frequency"].astype(float)
    logger.info("Loaded %d gnomAD variants", len(df))
    return df
