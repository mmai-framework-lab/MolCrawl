"""Metric selection helpers for MoleculeNet tasks.

The registry-backed metrics live in
``molcrawl.tasks.evaluation._base.metric_registry``.  Here we only choose
which metrics are meaningful for each task type.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry

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
    }
