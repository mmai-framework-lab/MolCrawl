"""Metric helpers for ProteinGym.

Primary: Spearman rank correlation between the model score and the
experimental DMS score. Also exposes Pearson, bootstrap 95 % CIs, and
(optionally) AUROC / AUPRC when a binary ``DMS_bin_score`` column is
present. ``score_distribution_stats`` produces per-class LL / score
summaries keyed by the binary label so readers can see whether the
ref-vs-mutant likelihood gap separates the functional / non-functional
halves independent of any threshold choice.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry

logger = logging.getLogger(__name__)


def correlation_metrics(
    y_true: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    return {
        "spearman": float(default_registry.compute("spearman", y_true, y_score)),
        "pearson": float(default_registry.compute("pearson", y_true, y_score)),
    }


def optional_binary_metrics(
    y_true_binary: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    if len(np.unique(y_true_binary)) < 2:
        return {}
    return {
        "auroc": float(default_registry.compute("auroc", y_true_binary, y_score)),
        "auprc": float(default_registry.compute("auprc", y_true_binary, y_score)),
    }


def bootstrap_correlation_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """Return percentile-based bootstrap ``(lo, hi)`` intervals for
    Spearman and Pearson over the aligned arrays."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    if n < 5 or n_boot <= 0:
        return {}

    rng = np.random.default_rng(seed)
    spearmans: List[float] = []
    pearsons: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if np.unique(yt).size < 2 or np.unique(ys).size < 2:
            continue
        spearmans.append(
            float(default_registry.compute("spearman", yt, ys))
        )
        pearsons.append(
            float(default_registry.compute("pearson", yt, ys))
        )
    if not spearmans:
        return {}
    lo_p = 100.0 * alpha / 2.0
    hi_p = 100.0 * (1.0 - alpha / 2.0)
    return {
        "spearman": (
            float(np.percentile(spearmans, lo_p)),
            float(np.percentile(spearmans, hi_p)),
        ),
        "pearson": (
            float(np.percentile(pearsons, lo_p)),
            float(np.percentile(pearsons, hi_p)),
        ),
    }


def score_distribution_stats(
    dms: np.ndarray,
    ref_ll: np.ndarray,
    mut_ll: np.ndarray,
    scores: np.ndarray,
    bin_labels: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Summarise the likelihood signal, per DMS-bin when available.

    When ``bin_labels`` is ``None`` returns a single ``"all"`` summary;
    when it carries binary class labels (0 / 1) returns separate blocks
    per class so the model's LL gap vs fitness is visible at the
    class level.
    """
    def _block(mask: np.ndarray) -> Dict[str, float]:
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "dms_mean": float(np.mean(dms[mask])),
            "dms_std": float(np.std(dms[mask])),
            "ll_ref_mean": float(np.mean(ref_ll[mask])),
            "ll_mut_mean": float(np.mean(mut_ll[mask])),
            "score_mean": float(np.mean(scores[mask])),
            "score_std": float(np.std(scores[mask])),
        }

    if bin_labels is None:
        return {"all": _block(np.ones(len(dms), dtype=bool))}

    bin_labels = np.asarray(bin_labels)
    out: Dict[str, Dict[str, float]] = {}
    unique = sorted({int(x) for x in np.unique(bin_labels) if not np.isnan(x)})
    for label in unique:
        mask = bin_labels == label
        name = "non_functional" if label == 0 else "functional" if label == 1 else f"bin_{label}"
        out[name] = _block(mask)
    return out
