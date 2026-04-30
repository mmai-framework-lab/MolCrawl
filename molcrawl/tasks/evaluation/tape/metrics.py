"""TAPE metric dispatch.

Picks the metric pack appropriate for each task type:

* classification / sequence_labeling: accuracy, F1 (macro), MCC
* regression: RMSE, Spearman, Pearson
* contact_prediction: placeholder ``precision_at_L_over_5`` that
  requires external CASP-style evaluation; the default implementation
  returns NaN and the real computation is wired in a follow-up PR.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": default_registry.compute("accuracy", y_true, y_pred),
        "f1_macro": default_registry.compute("f1_macro", y_true, y_pred),
        "mcc": default_registry.compute("mcc", y_true, y_pred),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": default_registry.compute("rmse", y_true, y_pred),
        "spearman": default_registry.compute("spearman", y_true, y_pred),
        "pearson": default_registry.compute("pearson", y_true, y_pred),
    }


def contact_prediction_metrics() -> Dict[str, float]:
    # Placeholder: the upstream protocol requires PDB-specific masking.
    return {"precision_at_L_over_5": float("nan")}
