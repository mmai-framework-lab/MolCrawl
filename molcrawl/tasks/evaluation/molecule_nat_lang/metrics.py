"""Likelihood-based metric pack for molecule/caption pairs."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np


def summarise(log_likelihoods: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(log_likelihoods), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {
            "mean_log_likelihood": float("nan"),
            "perplexity": float("nan"),
            "n_pairs_scored": 0,
        }
    mean_ll = float(arr.mean())
    return {
        "mean_log_likelihood": mean_ll,
        "perplexity": math.exp(-mean_ll),
        "n_pairs_scored": int(arr.size),
    }


def bootstrap_perplexity_ci(
    log_likelihoods: Sequence[float],
    n_boot: int = 100,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    arr = np.asarray(list(log_likelihoods), dtype=float)
    arr = arr[~np.isnan(arr)]
    if n_boot <= 0 or arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    n = arr.size
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        estimates[k] = math.exp(-float(arr[idx].mean()))
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))
