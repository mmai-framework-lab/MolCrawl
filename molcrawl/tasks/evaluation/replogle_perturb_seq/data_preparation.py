"""Replogle Perturb-seq loader.

Expected schema (CSV): ``perturbation`` (gene KO target), ``baseline``
(control expression vector, list / JSON), ``perturbed`` (post-KO
vector), and ``gene_names`` (optional list aligning to the vectors).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_replogle(path: Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = {"perturbation", "baseline", "perturbed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Replogle file missing required columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )

    def _parse(value: object):
        if isinstance(value, str):
            return json.loads(value)
        return value

    df["baseline"] = df["baseline"].map(_parse)
    df["perturbed"] = df["perturbed"].map(_parse)
    logger.info("Loaded %d perturbations", len(df))
    return df
