"""Metric helpers for ChEMBL scaffold held-out evaluation."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np


def perplexity_from_log_likelihoods(
    log_likelihoods: Sequence[float],
) -> float:
    """Convert per-sequence mean log-likelihoods into a single perplexity.

    The adapter reports mean per-token log-probabilities; this helper
    averages them across sequences and returns ``exp(-mean_ll)``.
    """
    arr = np.asarray(list(log_likelihoods), dtype=float)
    if arr.size == 0:
        return float("nan")
    mean_ll = float(arr.mean())
    return math.exp(-mean_ll)


def probe_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """AUROC / AUPRC / accuracy wrapper using the default registry."""
    from molcrawl.tasks.evaluation._base import default_registry

    out: Dict[str, float] = {}
    if len(np.unique(y_true)) >= 2:
        out["auroc"] = default_registry.compute("auroc", y_true, y_score)
        out["auprc"] = default_registry.compute("auprc", y_true, y_score)
    preds = (y_score >= 0.5).astype(int)
    out["accuracy"] = default_registry.compute("accuracy", y_true, preds)
    out["f1_binary"] = default_registry.compute("f1_binary", y_true, preds)
    return out
