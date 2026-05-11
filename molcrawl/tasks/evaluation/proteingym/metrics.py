"""Metric helpers for ProteinGym.

The canonical ProteinGym benchmark reports Spearman rank correlation
between the model score and the experimental DMS score; we additionally
expose Pearson, AUROC (binary label, when available) and AUPRC so
downstream dashboards can slice by metric family.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry


def correlation_metrics(
    y_true: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    out = {
        "spearman": default_registry.compute("spearman", y_true, y_score),
        "pearson": default_registry.compute("pearson", y_true, y_score),
    }
    return out


def optional_binary_metrics(
    y_true_binary: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    if len(np.unique(y_true_binary)) < 2:
        return {}
    return {
        "auroc": default_registry.compute("auroc", y_true_binary, y_score),
        "auprc": default_registry.compute("auprc", y_true_binary, y_score),
    }
