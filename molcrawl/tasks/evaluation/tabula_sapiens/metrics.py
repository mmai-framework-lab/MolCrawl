"""Multiclass cell-type metrics.

Labels may be arbitrary strings, so we factorise them before dispatching
to the integer-only metric registry entries.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry


def cell_type_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    mapping = {label: idx for idx, label in enumerate(classes)}
    y_true_int = np.array([mapping[v] for v in y_true_arr], dtype=int)
    y_pred_int = np.array([mapping[v] for v in y_pred_arr], dtype=int)
    return {
        "accuracy": default_registry.compute("accuracy", y_true_int, y_pred_int),
        "f1_macro": default_registry.compute("f1_macro", y_true_int, y_pred_int),
    }
