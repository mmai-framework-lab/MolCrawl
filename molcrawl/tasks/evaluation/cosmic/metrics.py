"""COSMIC metric helpers — threshold + bootstrap CI + score-distribution.

Reuses ClinVar's threshold/confusion helpers (since both tasks reduce to the
same LL(ref) − LL(var) → binary pathogenicity score) and adds:

- :func:`bootstrap_binary_ci` — 95 % CI for AUROC / AUPRC / accuracy / F1 /
  sensitivity / specificity via row-resampling. Mirrors gnomAD's
  ``bootstrap_correlation_ci`` and DeepLoc's ``bootstrap_multiclass_ci``.
- :func:`score_distribution_stats` — per-class summary of LL(ref), LL(var)
  and the score itself; the same shape ClinVar emits so the dashboard
  renders cosmic and clinvar in a consistent layout.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry
from molcrawl.tasks.evaluation.clinvar.metrics import (
    confusion_summary,
    find_optimal_f1_threshold,
    sensitivity_specificity,
)

logger = logging.getLogger(__name__)

__all__ = [
    "confusion_summary",
    "find_optimal_f1_threshold",
    "sensitivity_specificity",
    "bootstrap_binary_ci",
    "score_distribution_stats",
]


def bootstrap_binary_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
    n_boot: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """Return ``(lo, hi)`` 95 % bootstrap CIs for the binary metric pack.

    Resamples row indices with replacement ``n_boot`` times. Resamples that
    collapse to a single class (e.g. all-pathogenic) are skipped because
    AUROC / AUPRC are undefined there. When ``threshold`` is provided the
    threshold-based metrics (accuracy / f1 / sensitivity / specificity) are
    bootstrapped at the same cut so the CI reflects sampling variance only,
    not threshold-search variance.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n = labels.size
    if n < 5 or n_boot <= 0:
        return {}

    rng = np.random.default_rng(seed)
    auroc: List[float] = []
    auprc: List[float] = []
    acc: List[float] = []
    f1: List[float] = []
    sens: List[float] = []
    spec: List[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        s = scores[idx]
        if np.unique(y).size < 2:
            continue
        try:
            auroc.append(float(default_registry.compute("auroc", y, s)))
            auprc.append(float(default_registry.compute("auprc", y, s)))
        except Exception:
            continue
        if threshold is not None:
            preds = (s > threshold).astype(int)
            try:
                acc.append(float(default_registry.compute("accuracy", y, preds)))
                f1.append(float(default_registry.compute("f1_binary", y, preds)))
            except Exception:
                pass
            sens_b, spec_b = sensitivity_specificity(y, preds)
            sens.append(float(sens_b))
            spec.append(float(spec_b))

    if not auroc:
        return {}

    lo_p = 100.0 * alpha / 2.0
    hi_p = 100.0 * (1.0 - alpha / 2.0)

    def _ci(vals: List[float]) -> Tuple[float, float]:
        if not vals:
            return float("nan"), float("nan")
        return (
            float(np.percentile(vals, lo_p)),
            float(np.percentile(vals, hi_p)),
        )

    out: Dict[str, Tuple[float, float]] = {
        "auroc": _ci(auroc),
        "auprc": _ci(auprc),
    }
    if threshold is not None:
        out["accuracy"] = _ci(acc)
        out["f1_binary"] = _ci(f1)
        out["sensitivity"] = _ci(sens)
        out["specificity"] = _ci(spec)
    return out


def score_distribution_stats(
    labels: np.ndarray,
    ref_ll: np.ndarray,
    var_ll: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Per-class mean/std of LL(ref), LL(var), and the score.

    Same shape as the clinvar version so the dashboard renders both tasks
    with the same layout. Class names use the COSMIC convention
    (``pathogenic`` / ``neutral``) so the labels match what
    :mod:`prepare_csv` writes (``DAMAGING`` / ``NEUTRAL`` → cosmic_label
    1 / 0).
    """
    out: Dict[str, Dict[str, float]] = {}
    for label_value, name in ((0, "neutral"), (1, "pathogenic")):
        mask = labels == label_value
        n = int(mask.sum())
        if n == 0:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n": n,
            "ll_ref_mean": float(np.mean(ref_ll[mask])),
            "ll_ref_std": float(np.std(ref_ll[mask])),
            "ll_var_mean": float(np.mean(var_ll[mask])),
            "ll_var_std": float(np.std(var_ll[mask])),
            "score_mean": float(np.mean(scores[mask])),
            "score_std": float(np.std(scores[mask])),
        }
    return out
