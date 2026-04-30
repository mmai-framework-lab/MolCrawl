"""Task-type-specific metric dispatch for ChemLLMBench."""

from __future__ import annotations

from typing import Dict, Sequence

from molcrawl.tasks.evaluation.chebi20.metrics import smiles_metrics, text_metrics


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    if not predictions:
        return {"exact_match": 0.0}
    hits = sum(1 for p, r in zip(predictions, references) if str(p).strip() == str(r).strip())
    return {"exact_match": hits / len(predictions)}


def regression_metric(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    import numpy as np

    def _try_float(value: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    preds = np.array([_try_float(p) for p in predictions], dtype=float)
    refs = np.array([_try_float(r) for r in references], dtype=float)
    mask = ~(np.isnan(preds) | np.isnan(refs))
    if mask.sum() == 0:
        return {"rmse": float("nan"), "mae": float("nan")}
    diff = preds[mask] - refs[mask]
    return {
        "rmse": float((diff ** 2).mean() ** 0.5),
        "mae": float(abs(diff).mean()),
        "parse_rate": float(mask.sum() / len(preds)),
    }


def text_pack(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    return text_metrics(predictions, references)


def smiles_pack(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    return smiles_metrics(predictions, references)
