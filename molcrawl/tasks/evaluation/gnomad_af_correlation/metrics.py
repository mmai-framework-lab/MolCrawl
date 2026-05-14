"""Correlation metrics for gnomAD allele frequency."""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry

from .splits import DEFAULT_AF_BINS, assign_af_bins, bin_label

logger = logging.getLogger(__name__)


def correlation_metrics(af: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Plain Spearman / Pearson over the whole sample."""
    return {
        "spearman": float(default_registry.compute("spearman", af, scores)),
        "pearson": float(default_registry.compute("pearson", af, scores)),
    }


def bootstrap_correlation_ci(
    af: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """Return ``(lo, hi)`` bootstrap CIs for spearman and pearson.

    Draws ``n_boot`` resamples (with replacement) of the aligned
    ``(af, scores)`` pairs, recomputes each correlation on the resample,
    and reports the ``[alpha/2, 1 - alpha/2]`` percentile interval.
    """
    af_arr = np.asarray(af, dtype=float)
    score_arr = np.asarray(scores, dtype=float)
    n = len(af_arr)
    if n < 5 or n_boot <= 0:
        return {}

    rng = np.random.default_rng(seed)
    spearmans: List[float] = []
    pearsons: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        af_b = af_arr[idx]
        score_b = score_arr[idx]
        if np.unique(af_b).size < 2 or np.unique(score_b).size < 2:
            continue
        spearmans.append(
            float(default_registry.compute("spearman", af_b, score_b))
        )
        pearsons.append(
            float(default_registry.compute("pearson", af_b, score_b))
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


def per_bin_correlation(
    af: np.ndarray,
    scores: np.ndarray,
    bins: Sequence[Tuple[float, float]] = DEFAULT_AF_BINS,
) -> Dict[str, Dict[str, float]]:
    """Return per-AF-bin Spearman / Pearson + counts.

    Helps localize whether the model's signal sits on rare vs common
    variants, which is invisible in the pooled correlation alone.
    """
    af_arr = np.asarray(af, dtype=float)
    score_arr = np.asarray(scores, dtype=float)
    bin_ids = assign_af_bins(af_arr, bins)

    out: Dict[str, Dict[str, float]] = {}
    for i, (lo, hi) in enumerate(bins):
        mask = bin_ids == i
        n = int(mask.sum())
        label = bin_label(lo, hi)
        if n < 5 or np.unique(score_arr[mask]).size < 2:
            out[label] = {
                "n": n,
                "spearman": float("nan"),
                "pearson": float("nan"),
            }
            continue
        out[label] = {
            "n": n,
            "spearman": float(
                default_registry.compute("spearman", af_arr[mask], score_arr[mask])
            ),
            "pearson": float(
                default_registry.compute("pearson", af_arr[mask], score_arr[mask])
            ),
        }
    return out


def score_distribution_stats(
    af: np.ndarray,
    ref_ll: np.ndarray,
    var_ll: np.ndarray,
    scores: np.ndarray,
    bins: Sequence[Tuple[float, float]] = DEFAULT_AF_BINS,
) -> Dict[str, Dict[str, float]]:
    """Per-AF-bin summary statistics of the raw likelihood signal."""
    af_arr = np.asarray(af, dtype=float)
    bin_ids = assign_af_bins(af_arr, bins)

    out: Dict[str, Dict[str, float]] = {}
    for i, (lo, hi) in enumerate(bins):
        mask = bin_ids == i
        n = int(mask.sum())
        label = bin_label(lo, hi)
        if n == 0:
            out[label] = {"n": 0}
            continue
        out[label] = {
            "n": n,
            "ll_ref_mean": float(np.mean(ref_ll[mask])),
            "ll_ref_std": float(np.std(ref_ll[mask])),
            "ll_var_mean": float(np.mean(var_ll[mask])),
            "ll_var_std": float(np.std(var_ll[mask])),
            "score_mean": float(np.mean(scores[mask])),
            "score_std": float(np.std(scores[mask])),
        }
    return out
