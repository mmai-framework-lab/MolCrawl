"""Aggregate per-group perplexity + bootstrap CIs."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np


def perplexity_from_mean_nll(mean_nll: float) -> float:
    return math.exp(mean_nll)


def summarise_group(log_likelihoods: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(log_likelihoods), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {
            "perplexity": float("nan"),
            "mean_log_likelihood": float("nan"),
            "n": 0,
        }
    mean_ll = float(arr.mean())
    return {
        "mean_log_likelihood": mean_ll,
        "perplexity": perplexity_from_mean_nll(-mean_ll),
        "n": int(arr.size),
    }


def bootstrap_perplexity_ci(
    log_likelihoods: Sequence[float],
    n_boot: int = 100,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Resample sequences with replacement to bracket per-group perplexity.

    Returns ``(ci_lo, ci_hi)``. ``n_boot <= 0`` disables and returns
    ``(nan, nan)``.
    """
    arr = np.asarray(list(log_likelihoods), dtype=float)
    arr = arr[~np.isnan(arr)]
    if n_boot <= 0 or arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = arr.size
    estimates = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        estimates[k] = math.exp(-float(arr[idx].mean()))
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(estimates, alpha))
    hi = float(np.quantile(estimates, 1.0 - alpha))
    return lo, hi
