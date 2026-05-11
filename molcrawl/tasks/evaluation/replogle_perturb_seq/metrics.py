"""Per-perturbation Spearman / Pearson aggregation."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry


def delta_correlation(
    observed: Sequence[Sequence[float]], predicted: Sequence[Sequence[float]]
) -> Dict[str, float]:
    if len(observed) == 0 or len(predicted) == 0:
        return {"spearman_mean": float("nan"), "pearson_mean": float("nan")}
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
    return {
        "spearman_mean": float(np.mean(spearmans)) if spearmans else float("nan"),
        "pearson_mean": float(np.mean(pearsons)) if pearsons else float("nan"),
        "num_perturbations_scored": float(len(spearmans)),
    }
