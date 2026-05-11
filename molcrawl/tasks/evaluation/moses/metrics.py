"""MOSES-family distribution-learning metrics.

Core metrics (validity, uniqueness, novelty, internal diversity) are
delegated to :data:`molcrawl.tasks.evaluation._base.default_registry`.
This module also exposes thin shims for the optional FCD / SNN / Fragment /
Scaffold metrics that live in the upstream ``moses`` / ``fcd`` packages,
so the evaluator can report them when those dependencies are installed
without failing in minimal environments.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

from molcrawl.tasks.evaluation._base import default_registry

logger = logging.getLogger(__name__)


def distribution_metrics(
    generated: Sequence[str],
    reference: Sequence[str],
) -> Dict[str, float]:
    """Return the core MOSES metrics, skipping any that require RDKit if absent."""
    metrics: Dict[str, float] = {
        "validity": default_registry.compute("validity", generated),
        "uniqueness": default_registry.compute("uniqueness", generated),
        "novelty": default_registry.compute("novelty", generated, reference),
        "internal_diversity": default_registry.compute("internal_diversity", generated),
    }
    return metrics


def optional_extended_metrics(
    generated: Sequence[str],
    reference: Sequence[str],
) -> Optional[Dict[str, float]]:
    """Best-effort wrapper over the reference ``moses`` package.

    Returns ``None`` when the upstream dependency is unavailable, so the
    evaluator can continue with just the core metrics.
    """
    try:
        from moses.metrics import get_all_metrics  # type: ignore
    except ImportError:
        logger.info("Optional dependency 'moses' not available; skipping FCD / SNN / Fragment / Scaffold")
        return None

    try:
        extended = get_all_metrics(
            gen=list(generated),
            train=list(reference),
            n_jobs=1,
            device="cpu",
        )
    except Exception:  # pragma: no cover - upstream is heavy, best-effort only
        logger.exception("MOSES reference metric computation failed")
        return None

    # Flatten numeric entries only.
    out: Dict[str, float] = {}
    for key, value in extended.items():
        try:
            out[f"moses.{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return out
