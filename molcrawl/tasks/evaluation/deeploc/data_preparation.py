"""DeepLoc 2.0 loader + sampling helpers.

Expected columns (after :mod:`prepare_csv` reshaping):

- ``sequence``     — protein amino-acid sequence
- ``localisation`` — one of 10 canonical DeepLoc classes
- ``cluster_id``   — DeepLoc Partition id (1-5), used for cluster splits
- ``kingdom``      — taxonomic kingdom tag (Eukaryota / Bacteria / ...)
- ``accession``    — UniProt accession (passthrough, optional)

Raw upstream CSVs that have NOT been through ``prepare_csv`` (i.e. the
multi-label form) are still accepted as long as they expose ``sequence``
and ``localisation``; missing optional columns are tolerated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
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
    logger.info("Loaded %d DeepLoc sequences from %s", len(df), csv_path)
    return df


def stratified_subsample(
    df: pd.DataFrame,
    n_examples: int,
    label_column: str = "localisation",
    seed: int = 42,
) -> pd.DataFrame:
    """Class-balanced subsample so the test split sees every class.

    Reproducible given ``seed``. Falls back to uniform random sampling
    if the label column is missing, has fewer than 2 classes, or no
    examples are present.
    """
    if n_examples >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)

    if label_column in df.columns:
        pool = df.dropna(subset=[label_column])
        classes = sorted(pool[label_column].dropna().unique())
        if len(classes) >= 2:
            parts = []
            base = n_examples // len(classes)
            remainder = n_examples - base * len(classes)
            for i, cls in enumerate(classes):
                quota = base + (1 if i < remainder else 0)
                sub = pool[pool[label_column] == cls]
                take = min(quota, len(sub))
                state = int(rng.integers(0, 2**32 - 1))
                parts.append(sub.sample(n=take, random_state=state))
            sampled = pd.concat(parts, ignore_index=False)
            logger.info(
                "stratified_subsample: %d rows across %d classes",
                len(sampled),
                len(classes),
            )
            return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    state = int(rng.integers(0, 2**32 - 1))
    logger.info("stratified_subsample: uniform random %d rows (no label stratification)", n_examples)
    return df.sample(n=n_examples, random_state=state).reset_index(drop=True)
