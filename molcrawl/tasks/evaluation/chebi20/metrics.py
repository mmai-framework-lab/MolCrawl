"""Text + SMILES metrics for ChEBI-20.

BLEU / ROUGE / METEOR and SMILES exact-match / Levenshtein / validity.
All heavy-weight NLP dependencies (``nltk``, ``rouge-score``) are
imported lazily so the task skeleton can be tested in environments
without them.
"""

from __future__ import annotations

import logging
from typing import Dict, Sequence

logger = logging.getLogger(__name__)


def text_metrics(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    """Return BLEU / ROUGE / exact-match.  Missing deps fall through silently."""
    out: Dict[str, float] = {}
    out["exact_match"] = _exact_match(predictions, references)
    out.update(_bleu(predictions, references))
    out.update(_rouge(predictions, references))
    return out


def smiles_metrics(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {
        "exact_match": _exact_match(predictions, references),
        "mean_levenshtein": _mean_levenshtein(predictions, references),
        "validity": _smiles_validity(predictions),
    }
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _exact_match(pred: Sequence[str], ref: Sequence[str]) -> float:
    if not pred or not ref:
        return 0.0
    hits = sum(1 for p, r in zip(pred, ref) if str(p).strip() == str(r).strip())
    return hits / len(pred)


def _bleu(pred: Sequence[str], ref: Sequence[str]) -> Dict[str, float]:
    try:
        import nltk  # type: ignore  # noqa: F401
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        return {"bleu4": float("nan")}
    smoothie = SmoothingFunction().method1
    scores = []
    for p, r in zip(pred, ref):
        scores.append(
            sentence_bleu(
                [str(r).split()], str(p).split(), smoothing_function=smoothie
            )
        )
    return {"bleu4": float(sum(scores) / len(scores)) if scores else float("nan")}


def _rouge(pred: Sequence[str], ref: Sequence[str]) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except ImportError:
        return {"rougeL": float("nan")}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    values = []
    for p, r in zip(pred, ref):
        res = scorer.score(str(r), str(p))
        values.append(res["rougeL"].fmeasure)
    return {"rougeL": float(sum(values) / len(values)) if values else float("nan")}


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        return _levenshtein(b, a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insert = previous[j] + 1
            delete = current[j - 1] + 1
            substitute = previous[j - 1] + (0 if ca == cb else 1)
            current[j] = min(insert, delete, substitute)
        previous = current
    return previous[-1]


def _mean_levenshtein(pred: Sequence[str], ref: Sequence[str]) -> float:
    if not pred or not ref:
        return float("nan")
    total = sum(_levenshtein(str(p), str(r)) for p, r in zip(pred, ref))
    return total / len(pred)


def _smiles_validity(predictions: Sequence[str]) -> float:
    try:
        from rdkit import Chem
    except ImportError:
        return float("nan")
    if not predictions:
        return 0.0
    valid = 0
    for smi in predictions:
        if Chem.MolFromSmiles(str(smi)) is not None:
            valid += 1
    return valid / len(predictions)
