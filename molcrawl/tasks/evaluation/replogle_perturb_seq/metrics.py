"""Per-perturbation Spearman / Pearson aggregation + bootstrap CI."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry


def _per_pert_corr(
    observed: Sequence[Sequence[float]], predicted: Sequence[Sequence[float]]
):
    spearmans = []
    pearsons = []
    for obs, pred in zip(observed, predicted):
        obs_arr = np.asarray(obs, dtype=float)
        pred_arr = np.asarray(pred, dtype=float)
        if obs_arr.size < 2 or pred_arr.size < 2:
            continue
        if np.std(obs_arr) == 0 or np.std(pred_arr) == 0:
            continue
        spearmans.append(default_registry.compute("spearman", obs_arr, pred_arr))
        pearsons.append(default_registry.compute("pearson", obs_arr, pred_arr))
    return spearmans, pearsons


def delta_correlation(
    observed: Sequence[Sequence[float]], predicted: Sequence[Sequence[float]]
) -> Dict[str, float]:
    if len(observed) == 0 or len(predicted) == 0:
        return {
            "spearman_mean": float("nan"),
            "pearson_mean": float("nan"),
            "num_perturbations_scored": 0.0,
        }
    spearmans, pearsons = _per_pert_corr(observed, predicted)
    return {
        "spearman_mean": float(np.mean(spearmans)) if spearmans else float("nan"),
        "pearson_mean": float(np.mean(pearsons)) if pearsons else float("nan"),
        "num_perturbations_scored": float(len(spearmans)),
    }


def bootstrap_correlation_ci(
    observed: Sequence[Sequence[float]],
    predicted: Sequence[Sequence[float]],
    n_boot: int = 100,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap CIs for the per-perturbation mean Spearman / Pearson.

    Resamples *whole perturbations* (rows) with replacement so the CI
    reflects perturbation-level uncertainty, not gene-level. Skips when
    n_boot <= 0.
    """
    if n_boot <= 0:
        return {}
    obs = list(observed)
    pred = list(predicted)
    if not obs:
        return {}

    rng = np.random.default_rng(seed)
    spear_arr, pear_arr = [], []
    n = len(obs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ob = [obs[i] for i in idx]
        pr = [pred[i] for i in idx]
        s, p = _per_pert_corr(ob, pr)
        spear_arr.append(float(np.mean(s)) if s else float("nan"))
        pear_arr.append(float(np.mean(p)) if p else float("nan"))

    alpha = (1.0 - ci) / 2.0

    def _ci(arr) -> Tuple[float, float]:
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return float("nan"), float("nan")
        return float(np.quantile(a, alpha)), float(np.quantile(a, 1.0 - alpha))

    return {"spearman_mean": _ci(spear_arr), "pearson_mean": _ci(pear_arr)}
