"""Metric selection helpers for MoleculeNet tasks.

The registry-backed metrics live in
``molcrawl.tasks.evaluation._base.metric_registry``. This module chooses
which metrics are meaningful for each task type, plus provides a
bootstrap CI helper keyed on the primary ranking metric (AUROC for
classification, RMSE for regression).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation._base import default_registry

logger = logging.getLogger(__name__)

CLASSIFICATION_METRICS: Sequence[str] = ("auroc", "auprc", "accuracy", "f1_binary")
REGRESSION_METRICS: Sequence[str] = ("rmse", "mae", "r2")


def score_classification(
    y_true: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    threshold = 0.5
    y_pred = (y_score >= threshold).astype(int)
    out: Dict[str, float] = {}
    # AUROC / AUPRC need both classes present; degrade gracefully.
    if len(np.unique(y_true)) >= 2:
        out["auroc"] = default_registry.compute("auroc", y_true, y_score)
        out["auprc"] = default_registry.compute("auprc", y_true, y_score)
    out["accuracy"] = default_registry.compute("accuracy", y_true, y_pred)
    out["f1_binary"] = default_registry.compute("f1_binary", y_true, y_pred)
    return out


def score_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": default_registry.compute("rmse", y_true, y_pred),
        "mae": default_registry.compute("mae", y_true, y_pred),
        "r2": default_registry.compute("r2", y_true, y_pred),
        "spearman": default_registry.compute("spearman", y_true, y_pred),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task_type: str,
    n_boot: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """Return percentile ``(lo, hi)`` intervals for the primary metrics.

    Classification: auroc and auprc.
    Regression:     rmse and r2 and spearman.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if n < 5 or n_boot <= 0:
        return {}

    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = {}

    if task_type == "classification":
        if len(np.unique(y_true)) < 2:
            return {}
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yt = y_true[idx]
            ys = y_score[idx]
            if len(np.unique(yt)) < 2:
                continue
            samples.setdefault("auroc", []).append(
                float(default_registry.compute("auroc", yt, ys))
            )
            samples.setdefault("auprc", []).append(
                float(default_registry.compute("auprc", yt, ys))
            )
    else:
        yt_num = np.asarray(y_true, dtype=float)
        ys_num = np.asarray(y_score, dtype=float)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yt = yt_num[idx]
            ys = ys_num[idx]
            if np.std(yt) == 0 or np.std(ys) == 0:
                continue
            samples.setdefault("rmse", []).append(
                float(default_registry.compute("rmse", yt, ys))
            )
            samples.setdefault("r2", []).append(
                float(default_registry.compute("r2", yt, ys))
            )
            samples.setdefault("spearman", []).append(
                float(default_registry.compute("spearman", yt, ys))
            )

    if not samples:
        return {}
    lo_p = 100.0 * alpha / 2.0
    hi_p = 100.0 * (1.0 - alpha / 2.0)
    return {
        k: (float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p)))
        for k, v in samples.items()
        if v
    }


def split_label_distribution(
    y: np.ndarray, task_type: str
) -> Dict[str, float]:
    """Return a concise label-distribution diagnostic for a split."""
    series = pd.Series(y)
    valid = series.dropna().to_numpy()
    if valid.size == 0:
        return {"n": 0}
    out: Dict[str, float] = {"n": int(valid.size)}
    if task_type == "classification":
        try:
            ints = valid.astype(int)
        except (TypeError, ValueError):
            return {"n": int(valid.size)}
        uniq, counts = np.unique(ints, return_counts=True)
        for u, c in zip(uniq, counts):
            out[f"class_{int(u)}_count"] = int(c)
        if 1 in uniq:
            out["positive_ratio"] = float(
                counts[list(uniq).index(1)] / valid.size
            )
    else:
        arr_f = valid.astype(float)
        out.update(
            {
                "mean": float(arr_f.mean()),
                "std": float(arr_f.std()),
                "min": float(arr_f.min()),
                "max": float(arr_f.max()),
                "median": float(np.median(arr_f)),
            }
        )
    return out
