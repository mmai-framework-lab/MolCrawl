"""Metric pack for DeepLoc subcellular localisation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": default_registry.compute("accuracy", y_true, y_pred),
        "f1_macro": default_registry.compute("f1_macro", y_true, y_pred),
        "mcc": default_registry.compute("mcc", y_true, y_pred),
    }


def bootstrap_multiclass_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 100,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap CIs for accuracy / f1_macro / mcc.

    Skips a metric when a resample collapses to one class (mcc / f1_macro
    are undefined). Returns NaN tuples in that case.
    """
    if n_boot <= 0:
        return {}
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    if yt.size == 0:
        return {}
    rng = np.random.default_rng(seed)

    acc_arr, f1_arr, mcc_arr = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, yt.size, size=yt.size)
        try:
            m = multiclass_metrics(yt[idx], yp[idx])
        except Exception:
            continue
        acc_arr.append(m.get("accuracy", float("nan")))
        f1_arr.append(m.get("f1_macro", float("nan")))
        mcc_arr.append(m.get("mcc", float("nan")))

    alpha = (1.0 - ci) / 2.0

    def _ci(arr) -> Tuple[float, float]:
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return float("nan"), float("nan")
        return float(np.quantile(a, alpha)), float(np.quantile(a, 1.0 - alpha))

    return {
        "accuracy": _ci(acc_arr),
        "f1_macro": _ci(f1_arr),
        "mcc": _ci(mcc_arr),
    }
