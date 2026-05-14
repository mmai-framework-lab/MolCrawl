"""TAPE metric dispatch + bootstrap CIs.

Metric packs:

* classification / sequence_labeling: accuracy, macro F1, MCC
* regression: RMSE, Spearman, Pearson
* contact_prediction: placeholder ``precision_at_L_over_5`` that
  requires external CASP-style evaluation; the default implementation
  returns NaN and the real computation is wired in a follow-up PR.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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


def contact_prediction_metrics(
    per_protein: list,
    min_separation: int = 24,
    ks: tuple = (1, 2, 5),
) -> Dict[str, float]:
    """Long-range precision@L/k for contact maps.

    ``per_protein`` is a list of dicts with keys::

        seq_len   : int — protein length L
        pair_idx  : np.ndarray of shape (P, 2) listing the (i, j) pairs
                    that the model SCORED (only long-range pairs with
                    j - i >= min_separation should be passed in).
        scores    : np.ndarray of shape (P,) — predicted contact scores
                    aligned with ``pair_idx``.
        labels    : np.ndarray of shape (P,) of 0/1 ground-truth contact
                    labels for the same pairs.

    For each protein, sort scored pairs by score (desc), take the top
    ``L // k`` for each ``k`` in ``ks``, and report the fraction that
    are true contacts. Returns the mean across proteins, the median,
    and the per-k count of proteins scored. NaN when no protein has
    enough long-range pairs to evaluate at that k.
    """
    if not per_protein:
        return {f"precision_at_L_over_{k}": float("nan") for k in ks} | {
            "n_proteins_scored": 0.0,
        }

    out: Dict[str, float] = {}
    n_scored = 0
    per_k: Dict[int, list] = {k: [] for k in ks}
    for prot in per_protein:
        L = int(prot["seq_len"])
        pair_idx = np.asarray(prot["pair_idx"], dtype=int)
        scores = np.asarray(prot["scores"], dtype=float)
        labels = np.asarray(prot["labels"], dtype=int)
        # filter to long-range pairs (defensive — caller should already pass these)
        if pair_idx.size > 0:
            sep = pair_idx[:, 1] - pair_idx[:, 0]
            keep = sep >= min_separation
            scores = scores[keep]
            labels = labels[keep]
        if scores.size == 0:
            continue
        order = np.argsort(-scores)
        scored = False
        for k in ks:
            top_n = max(1, L // k)
            if scores.size < top_n:
                continue
            top_idx = order[:top_n]
            precision = float(labels[top_idx].mean())
            per_k[k].append(precision)
            scored = True
        if scored:
            n_scored += 1

    for k in ks:
        vals = per_k[k]
        if vals:
            out[f"precision_at_L_over_{k}"] = float(np.mean(vals))
            out[f"precision_at_L_over_{k}_median"] = float(np.median(vals))
        else:
            out[f"precision_at_L_over_{k}"] = float("nan")
            out[f"precision_at_L_over_{k}_median"] = float("nan")
    out["n_proteins_scored"] = float(n_scored)
    return out


def bootstrap_contact_ci(
    per_protein: list,
    min_separation: int = 24,
    ks: tuple = (1, 2, 5),
    n_boot: int = 100,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, "tuple[float, float]"]:
    """Bootstrap CIs for ``precision_at_L_over_k`` by resampling proteins."""
    if n_boot <= 0 or not per_protein:
        return {}
    n = len(per_protein)
    rng = np.random.default_rng(seed)
    buckets: Dict[str, list] = {f"precision_at_L_over_{k}": [] for k in ks}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            m = contact_prediction_metrics(
                [per_protein[i] for i in idx],
                min_separation=min_separation,
                ks=ks,
            )
        except Exception:
            continue
        for k in ks:
            key = f"precision_at_L_over_{k}"
            v = m.get(key)
            if v is not None and not np.isnan(v):
                buckets[key].append(float(v))

    alpha = (1.0 - ci) / 2.0

    def _ci(arr):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return float("nan"), float("nan")
        return float(np.quantile(a, alpha)), float(np.quantile(a, 1.0 - alpha))

    return {k: _ci(v) for k, v in buckets.items()}


def sequence_labeling_metrics(
    per_protein_pred: list,
    per_protein_label: list,
    per_protein_mask: list,
) -> Dict[str, float]:
    """Per-residue Q3/Q8 + macro F1 + per-protein mean accuracy.

    All inputs are lists of length ``n_proteins``; each entry is a
    1-D array (or list) of equal length per protein. Residues with
    ``mask[i] == 0`` are excluded from both pooled and per-protein scores.
    """
    if not per_protein_pred:
        return {
            "q_overall": float("nan"),
            "q_per_protein_mean": float("nan"),
            "f1_macro": float("nan"),
            "n_proteins_scored": 0.0,
            "n_residues_scored": 0.0,
        }

    flat_yt: list = []
    flat_yp: list = []
    per_prot_acc: list = []
    for preds, labels, mask in zip(per_protein_pred, per_protein_label, per_protein_mask):
        preds = np.asarray(preds, dtype=int)
        labels = np.asarray(labels, dtype=int)
        mask = np.asarray(mask, dtype=int)
        if preds.size == 0 or labels.size == 0:
            continue
        valid = mask.astype(bool)
        if not valid.any():
            continue
        p = preds[valid]
        y = labels[valid]
        flat_yt.extend(int(v) for v in y)
        flat_yp.extend(int(v) for v in p)
        per_prot_acc.append(float((p == y).mean()))

    if not flat_yt:
        return {
            "q_overall": float("nan"),
            "q_per_protein_mean": float("nan"),
            "f1_macro": float("nan"),
            "n_proteins_scored": 0.0,
            "n_residues_scored": 0.0,
        }
    yt_arr = np.asarray(flat_yt, dtype=int)
    yp_arr = np.asarray(flat_yp, dtype=int)
    return {
        "q_overall": float((yt_arr == yp_arr).mean()),
        "q_per_protein_mean": float(np.mean(per_prot_acc)) if per_prot_acc else float("nan"),
        "f1_macro": default_registry.compute("f1_macro", yt_arr, yp_arr),
        "n_proteins_scored": float(len(per_prot_acc)),
        "n_residues_scored": float(yt_arr.size),
    }


def bootstrap_sequence_labeling_ci(
    per_protein_pred: list,
    per_protein_label: list,
    per_protein_mask: list,
    n_boot: int = 100,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, "tuple[float, float]"]:
    """Resample whole proteins (rows), recompute Q-overall / per-protein mean / f1_macro."""
    if n_boot <= 0:
        return {}
    n = len(per_protein_pred)
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    keys = ("q_overall", "q_per_protein_mean", "f1_macro")
    buckets: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            m = sequence_labeling_metrics(
                [per_protein_pred[i] for i in idx],
                [per_protein_label[i] for i in idx],
                [per_protein_mask[i] for i in idx],
            )
        except Exception:
            continue
        for k in keys:
            v = m.get(k)
            if v is not None:
                buckets[k].append(float(v))

    alpha = (1.0 - ci) / 2.0

    def _ci(arr):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return float("nan"), float("nan")
        return float(np.quantile(a, alpha)), float(np.quantile(a, 1.0 - alpha))

    return {k: _ci(v) for k, v in buckets.items()}


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    n_boot: int = 100,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap CIs for the active metric pack.

    Returns ``{}`` for placeholders and when n_boot <= 0. Skips a metric
    when a resample collapses to one class (mcc / f1 are undefined).
    """
    if n_boot <= 0:
        return {}
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.size == 0:
        return {}

    rng = np.random.default_rng(seed)
    if task_type == "regression":
        keys = ("rmse", "spearman", "pearson")

        def compute(yti, ypi):
            return regression_metrics(yti.astype(float), ypi.astype(float))
    else:
        keys = ("accuracy", "f1_macro", "mcc")

        def compute(yti, ypi):
            return classification_metrics(yti.astype(int), ypi.astype(int))

    buckets: Dict[str, list] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, yt.size, size=yt.size)
        try:
            m = compute(yt[idx], yp[idx])
        except Exception:
            continue
        for k in keys:
            if k in m and m[k] is not None:
                buckets[k].append(m[k])

    alpha = (1.0 - ci) / 2.0

    def _ci(arr) -> Tuple[float, float]:
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return float("nan"), float("nan")
        return float(np.quantile(a, alpha)), float(np.quantile(a, 1.0 - alpha))

    return {k: _ci(v) for k, v in buckets.items()}
