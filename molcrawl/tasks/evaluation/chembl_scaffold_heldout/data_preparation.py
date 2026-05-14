"""Loader + sampling helpers for the ChEMBL scaffold held-out split.

The held-out CSV is produced by ``prepare_csv.py`` (scaffold-disjoint
split of ChEMBL's flat ``smiles.txt``). Any CSV with at least a
``smiles`` column (and optionally a label column for probe mode)
also works as input.

In addition to the basic CSV loader, this module exposes a
length-stratified subsample helper. The legacy path was
``df.head(max_examples)`` which silently biases toward whichever
ordering the file happens to use; for ChEMBL that bias is real
(the upstream pipeline groups by chembl_id, which clusters
analogues together).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
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
    logger.info("Loaded %d ChEMBL scaffold held-out examples from %s", len(df), csv_path)
    return df


def stratified_subsample(
    df: pd.DataFrame,
    n_examples: int,
    smiles_column: str = "smiles",
    label_column: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample ``n_examples`` rows with reproducible stratification.

    * Probe mode (``label_column`` given): class-balanced sample over the
      first (binary) label column. Falls back to uniform random if the
      label has fewer than 2 distinct values.
    * Perplexity mode (no label): SMILES-length quantile binning so the
      sample covers the short ↔ long axis. ChEMBL has a long tail of
      very long natural-product-style SMILES; ``df.head`` would over-
      represent the early-id (short) part of the corpus.
    """
    if n_examples >= len(df):
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)

    if label_column is not None and label_column in df.columns:
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
                "stratified_subsample (probe): %d rows across %d classes",
                len(sampled),
                len(classes),
            )
            return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Perplexity mode — bin by SMILES length.
    lengths = df[smiles_column].astype(str).str.len().to_numpy()
    n_bins = min(10, max(2, n_examples // 25))
    try:
        bin_ids = pd.qcut(
            lengths,
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
    except Exception:
        bin_ids = np.zeros(len(df), dtype=int)
    bin_arr = np.asarray(bin_ids)
    parts = []
    unique_bins = sorted(np.unique(bin_arr[~pd.isna(bin_arr)]))
    base = n_examples // max(1, len(unique_bins))
    remainder = n_examples - base * len(unique_bins)
    for i, b in enumerate(unique_bins):
        quota = base + (1 if i < remainder else 0)
        sub = df[bin_arr == b]
        take = min(quota, len(sub))
        state = int(rng.integers(0, 2**32 - 1))
        parts.append(sub.sample(n=take, random_state=state))
    sampled = pd.concat(parts, ignore_index=False)
    logger.info(
        "stratified_subsample (length): %d rows across %d length bins",
        len(sampled),
        len(unique_bins),
    )
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
