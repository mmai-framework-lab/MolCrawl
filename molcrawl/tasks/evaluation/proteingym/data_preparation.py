"""ProteinGym table loader.

Supports the DMS substitution subset.  Expected columns:

* ``mutated_sequence``  - variant sequence as a string
* ``wildtype_sequence`` - reference sequence
* ``DMS_score``         - experimental fitness label (float)
* ``DMS_bin_score`` (optional) - binarised label, used when the task is
  framed as classification
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_proteingym(path: Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = {"mutated_sequence", "wildtype_sequence", "DMS_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"ProteinGym file missing required columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )
    df = df.dropna(subset=list(required)).reset_index(drop=True)
    logger.info("Loaded %d ProteinGym variants from %s", len(df), csv_path)
    return df
