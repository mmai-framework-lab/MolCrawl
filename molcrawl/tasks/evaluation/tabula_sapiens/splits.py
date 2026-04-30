"""Cross-tissue / random splits for Tabula Sapiens."""

from __future__ import annotations

import numpy as np
from typing import Sequence, Tuple


def random_split(
    n: int, test_fraction: float = 0.2, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(n * (1 - test_fraction))
    return idx[:split], idx[split:]


def cross_tissue_split(
    tissues: Sequence[str], holdout_tissues: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    holdout = {t.lower() for t in holdout_tissues}
    tissue_arr = np.array([t.lower() for t in tissues])
    mask = np.isin(tissue_arr, list(holdout))
    train_idx = np.flatnonzero(~mask)
    test_idx = np.flatnonzero(mask)
    return train_idx, test_idx
