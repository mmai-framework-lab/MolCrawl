"""Multiclass cell-type metrics + bootstrap CI.

Labels may be arbitrary strings, so we factorise them before dispatching
to the integer-only metric registry entries.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry


def _factorise(y_true: np.ndarray, y_pred: np.ndarray):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    mapping = {label: idx for idx, label in enumerate(classes)}
    yt = np.array([mapping[v] for v in y_true], dtype=int)
    yp = np.array([mapping[v] for v in y_pred], dtype=int)
    return yt, yp


def cell_type_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt, yp = _factorise(np.asarray(y_true), np.asarray(y_pred))
    return {
        "accuracy": default_registry.compute("accuracy", yt, yp),
        "f1_macro": default_registry.compute("f1_macro", yt, yp),
    }


def bootstrap_celltype_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 100,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    if n_boot <= 0:
        return {}
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.size == 0:
        return {}

    rng = np.random.default_rng(seed)
    acc_arr, f1_arr = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, yt.size, size=yt.size)
        try:
            m = cell_type_metrics(yt[idx], yp[idx])
        except Exception:
            continue
        acc_arr.append(m.get("accuracy", float("nan")))
        f1_arr.append(m.get("f1_macro", float("nan")))

    alpha = (1.0 - ci) / 2.0

    def _ci(arr) -> Tuple[float, float]:
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return float("nan"), float("nan")
        return float(np.quantile(a, alpha)), float(np.quantile(a, 1.0 - alpha))

    return {"accuracy": _ci(acc_arr), "f1_macro": _ci(f1_arr)}
