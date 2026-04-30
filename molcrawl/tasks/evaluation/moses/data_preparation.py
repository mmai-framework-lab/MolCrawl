"""Reference-set loading for MOSES.

The MOSES benchmark distributes three canonical CSVs: ``train.csv``,
``test.csv``, and ``test_scaffolds.csv`` (ZINC-derived).  The evaluator
only needs the training SMILES (for novelty) and optionally the test
SMILES for downstream distribution metrics.  ``manifest.json`` next to
these files records provenance and SHA-256.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def load_reference_smiles(path: Path, column: str = "SMILES", limit: int | None = None) -> List[str]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if column not in df.columns:
        # Some distributions use lowercase or a single-column file.
        if "smiles" in df.columns:
            column = "smiles"
        else:
            raise ValueError(
                f"Reference CSV missing SMILES column (looked for {column!r}); "
                f"available: {list(df.columns)}"
            )
    smiles = df[column].astype(str).tolist()
    if limit is not None:
        smiles = smiles[: int(limit)]
    logger.info("Loaded %d reference SMILES from %s", len(smiles), csv_path)
    return smiles
