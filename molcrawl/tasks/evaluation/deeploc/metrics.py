"""Metric pack for DeepLoc subcellular localisation."""

from __future__ import annotations

from typing import Dict

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": default_registry.compute("accuracy", y_true, y_pred),
        "f1_macro": default_registry.compute("f1_macro", y_true, y_pred),
        "mcc": default_registry.compute("mcc", y_true, y_pred),
    }
