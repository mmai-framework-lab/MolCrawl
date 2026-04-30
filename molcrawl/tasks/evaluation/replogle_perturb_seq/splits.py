"""Replogle split by perturbation target."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def perturbation_split(
    df: pd.DataFrame, test_fraction: float = 0.2, seed: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    unique = df["perturbation"].unique().tolist()
    rng.shuffle(unique)
    cutoff = int(len(unique) * (1 - test_fraction))
    train_perts = set(unique[:cutoff])
    train_df = df[df["perturbation"].isin(train_perts)].reset_index(drop=True)
    test_df = df[~df["perturbation"].isin(train_perts)].reset_index(drop=True)
    return train_df, test_df
