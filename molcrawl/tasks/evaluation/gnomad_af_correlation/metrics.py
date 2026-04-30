"""Correlation metrics for gnomAD allele frequency."""

from __future__ import annotations

from typing import Dict

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry


def correlation_metrics(af: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "spearman": default_registry.compute("spearman", af, scores),
        "pearson": default_registry.compute("pearson", af, scores),
    }
