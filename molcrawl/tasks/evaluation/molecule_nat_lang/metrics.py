"""Likelihood-based metric pack for molecule/caption pairs."""

from __future__ import annotations

import math
from typing import Dict, Sequence


def summarise(log_likelihoods: Sequence[float]) -> Dict[str, float]:
    arr = list(log_likelihoods)
    if not arr:
        return {"mean_log_likelihood": float("nan"), "perplexity": float("nan")}
    mean_ll = sum(arr) / len(arr)
    return {
        "mean_log_likelihood": float(mean_ll),
        "perplexity": math.exp(-mean_ll),
    }
