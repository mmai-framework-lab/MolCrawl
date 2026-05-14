"""Metric pack for GUE sub-tasks."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, float]:
    metrics = {
        "accuracy": default_registry.compute("accuracy", y_true, y_pred),
        "f1_macro": default_registry.compute("f1_macro", y_true, y_pred),
    }
    if num_classes == 2:
        metrics["mcc"] = default_registry.compute("mcc", y_true, y_pred)
        metrics["f1_binary"] = default_registry.compute("f1_binary", y_true, y_pred)
    return metrics


def bootstrap_classification_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    n_boot: int = 100,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap CIs for the active metric pack.

    Skips a metric when a resample collapses to one class (mcc / f1_binary
    are undefined). Returns NaN tuples in that case.
    """
    if n_boot <= 0:
        return {}
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    if yt.size == 0:
        return {}

    rng = np.random.default_rng(seed)
    keys = ["accuracy", "f1_macro"]
    if num_classes == 2:
        keys += ["mcc", "f1_binary"]

    buckets: Dict[str, list] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, yt.size, size=yt.size)
        try:
            m = classification_metrics(yt[idx], yp[idx], num_classes)
        except Exception:
            continue
        for k in keys:
            if k in m:
                buckets[k].append(m[k])

    alpha = (1.0 - ci) / 2.0

    def _ci(arr) -> Tuple[float, float]:
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return float("nan"), float("nan")
        return float(np.quantile(a, alpha)), float(np.quantile(a, 1.0 - alpha))

    return {k: _ci(v) for k, v in buckets.items()}
