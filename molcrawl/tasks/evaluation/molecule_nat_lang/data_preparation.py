"""Loader + sampling helpers for molecule-caption pair evaluations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_pairs(
    path: Path,
    smiles_column: str = "smiles",
    caption_column: str = "caption",
) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    df = pd.read_csv(file_path)
    required = {smiles_column, caption_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"molecule_nat_lang file missing columns {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )
    df = df.dropna(subset=list(required)).reset_index(drop=True)
    logger.info("Loaded %d molecule-caption pairs from %s", len(df), file_path)
    return df


def stratified_subsample(
    df: pd.DataFrame,
    n_examples: int,
    smiles_column: str = "smiles",
    caption_column: str = "caption",
    seed: int = 42,
) -> pd.DataFrame:
    """Down-sample by combined-length quantile bins.

    The metric here is mean PLL of ``caption + smiles``; samples that
    happen to be all very short (or all very long) bias the perplexity
    point estimate. Bin by total length and sample proportionally so the
    perplexity reflects the corpus length distribution.
    """
    if n_examples >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)

    lengths = (
        df[smiles_column].astype(str).str.len()
        + df[caption_column].astype(str).str.len()
    ).to_numpy()
    n_bins = min(10, max(2, n_examples // 25))
    try:
        bin_ids = pd.qcut(
            lengths, q=n_bins, labels=False, duplicates="drop"
        ).to_numpy()
    except Exception:
        bin_ids = np.zeros(len(df), dtype=int)

    parts = []
    unique_bins = sorted(np.unique(bin_ids[~pd.isna(bin_ids)]))
    base = n_examples // max(1, len(unique_bins))
    remainder = n_examples - base * len(unique_bins)
    for i, b in enumerate(unique_bins):
        quota = base + (1 if i < remainder else 0)
        sub = df[bin_ids == b]
        take = min(quota, len(sub))
        state = int(rng.integers(0, 2**32 - 1))
        parts.append(sub.sample(n=take, random_state=state))
    sampled = pd.concat(parts, ignore_index=False)
    logger.info(
        "stratified_subsample: %d rows across %d combined-length bins",
        len(sampled),
        len(unique_bins),
    )
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
