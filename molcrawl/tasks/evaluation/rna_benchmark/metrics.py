"""Aggregate per-group perplexity + accuracy."""

from __future__ import annotations

import math
from typing import Dict, Sequence


def perplexity_from_mean_nll(mean_nll: float) -> float:
    return math.exp(mean_nll)


def summarise_group(log_likelihoods: Sequence[float]) -> Dict[str, float]:
    if not log_likelihoods:
        return {"perplexity": float("nan"), "mean_log_likelihood": float("nan")}
    mean_ll = sum(log_likelihoods) / len(log_likelihoods)
    return {
        "mean_log_likelihood": float(mean_ll),
        "perplexity": perplexity_from_mean_nll(-mean_ll),
    }
