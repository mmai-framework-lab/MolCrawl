"""ClinVar-specific metric helpers.

The standard binary classification metrics (accuracy, F1, AUROC, AUPRC)
live in :mod:`molcrawl.tasks.evaluation._base.metric_registry`.  This
module adds the threshold search + confusion-matrix pack used for
pathogenicity scoring.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def find_optimal_f1_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> float:
    """Return the score threshold that maximises binary F1.

    Enumerates candidate thresholds as midpoints between successive
    unique scores plus sentinels below the minimum and above the
    maximum.  This is more thorough than ``sklearn.metrics.roc_curve``
    (which drops intermediate thresholds) and is required to recover a
    perfect split when one exists.
    """
    from sklearn.metrics import precision_recall_fscore_support

    unique_scores = np.unique(scores)
    if unique_scores.size == 0:
        return 0.0

    candidates = [float(unique_scores[0]) - 1.0]
    if unique_scores.size > 1:
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        candidates.extend(float(m) for m in midpoints)
    candidates.append(float(unique_scores[-1]) + 1.0)

    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        predictions = (scores > threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def confusion_summary(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, int]:
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    cm = confusion_summary(y_true, y_pred)
    tp = cm["true_positive"]
    tn = cm["true_negative"]
    fp = cm["false_positive"]
    fn = cm["false_negative"]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity
