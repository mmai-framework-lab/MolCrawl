"""Metric helpers for ChEMBL scaffold held-out evaluation."""

from __future__ import annotations

import logging
import math
from typing import Dict, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def perplexity_from_log_likelihoods(
    log_likelihoods: Sequence[float],
) -> float:
    """Convert per-sequence mean log-likelihoods into a single perplexity.

    The adapter reports mean per-token log-probabilities; this helper
    averages them across sequences and returns ``exp(-mean_ll)``.
    """
    arr = np.asarray(list(log_likelihoods), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    mean_ll = float(arr.mean())
    return math.exp(-mean_ll)


def probe_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """AUROC / AUPRC / accuracy / F1 wrapper using the default registry."""
    from molcrawl.tasks.evaluation._base import default_registry

    out: Dict[str, float] = {}
    if len(np.unique(y_true)) >= 2:
        out["auroc"] = default_registry.compute("auroc", y_true, y_score)
        out["auprc"] = default_registry.compute("auprc", y_true, y_score)
    preds = (y_score >= 0.5).astype(int)
    out["accuracy"] = default_registry.compute("accuracy", y_true, preds)
    out["f1_binary"] = default_registry.compute("f1_binary", y_true, preds)
    return out


def bootstrap_perplexity_ci(
    log_likelihoods: Sequence[float],
    n_boot: int = 100,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Resample sequences with replacement to bracket perplexity uncertainty.

    Returns ``(ci_lo, ci_hi)``. ``n_boot <= 0`` disables and returns
    ``(nan, nan)`` so the caller can elide the field cheaply.
    """
    arr = np.asarray(list(log_likelihoods), dtype=float)
    arr = arr[~np.isnan(arr)]
    if n_boot <= 0 or arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    n = arr.size
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        estimates[k] = math.exp(-float(arr[idx].mean()))
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(estimates, alpha))
    hi = float(np.quantile(estimates, 1.0 - alpha))
    return lo, hi


def bootstrap_probe_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 100,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap CIs for AUROC / AUPRC / accuracy / F1.

    Skips a metric when a resample collapses to one class (AUROC/AUPRC
    are undefined there). Returns NaN tuples when no metric is computable.
    """
    if n_boot <= 0:
        return {}
    yt = np.asarray(y_true)
    ys = np.asarray(y_score)
    mask = ~np.isnan(yt.astype(float)) & ~np.isnan(ys.astype(float))
    yt = yt[mask].astype(int)
    ys = ys[mask].astype(float)
    if yt.size == 0:
        return {}

    rng = np.random.default_rng(seed)
    auroc_arr, auprc_arr, acc_arr, f1_arr = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, yt.size, size=yt.size)
        yt_b, ys_b = yt[idx], ys[idx]
        try:
            m = probe_metrics(yt_b, ys_b)
        except Exception:
            continue
        if "auroc" in m:
            auroc_arr.append(m["auroc"])
        if "auprc" in m:
            auprc_arr.append(m["auprc"])
        acc_arr.append(m.get("accuracy", float("nan")))
        f1_arr.append(m.get("f1_binary", float("nan")))

    alpha = (1.0 - ci) / 2.0

    def _ci(arr):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return float("nan"), float("nan")
        return float(np.quantile(a, alpha)), float(np.quantile(a, 1.0 - alpha))

    out: Dict[str, Tuple[float, float]] = {}
    if auroc_arr:
        out["auroc"] = _ci(auroc_arr)
    if auprc_arr:
        out["auprc"] = _ci(auprc_arr)
    out["accuracy"] = _ci(acc_arr)
    out["f1_binary"] = _ci(f1_arr)
    return out


def length_stats(smiles: Sequence[str]) -> Dict[str, float]:
    if not smiles:
        return {}
    lengths = np.array([len(s) for s in smiles], dtype=float)
    return {
        "smiles_length_mean": float(lengths.mean()),
        "smiles_length_std": float(lengths.std()),
        "smiles_length_min": float(lengths.min()),
        "smiles_length_median": float(np.median(lengths)),
        "smiles_length_max": float(lengths.max()),
    }
