"""Cross-tissue / random / precomputed splits for Tabula Sapiens."""

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


def precomputed_split(
    labels: Sequence[str], test_label: str = "test"
) -> Tuple[np.ndarray, np.ndarray]:
    """Use the split each row already carries.

    A random split over cells puts the same donor on both sides, and a linear
    probe can then read donor-specific signal instead of cell type. Which donors
    go where is a decision made once, against the cell counts and the assay and
    tissue balance, and recorded -- not redrawn per run. When the JSONL carries
    that decision, this returns it unchanged.
    """
    arr = np.asarray([str(x) for x in labels])
    test_idx = np.flatnonzero(arr == test_label)
    train_idx = np.flatnonzero(arr != test_label)
    return train_idx, test_idx
