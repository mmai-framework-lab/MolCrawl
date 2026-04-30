"""Metric pack for GUE sub-tasks."""

from __future__ import annotations

from typing import Dict

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
