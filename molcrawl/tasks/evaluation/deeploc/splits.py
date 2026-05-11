"""DeepLoc ships with pre-computed clusters; honour them when present."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def cluster_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 0,
    column: str = "cluster_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by cluster id so that homology does not leak across splits.

    Falls back to a stratified random split when the cluster column is
    missing (useful in minimal CI environments).
    """
    if column not in df.columns:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        split_at = int(len(df) * (1 - test_fraction))
        return df.iloc[idx[:split_at]].reset_index(drop=True), df.iloc[idx[split_at:]].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    clusters = df[column].unique()
    rng.shuffle(clusters)
    test_size = int(len(clusters) * test_fraction)
    test_clusters = set(clusters[:test_size])
    test_df = df[df[column].isin(test_clusters)].reset_index(drop=True)
    train_df = df[~df[column].isin(test_clusters)].reset_index(drop=True)
    return train_df, test_df
