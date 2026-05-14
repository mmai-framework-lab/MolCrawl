"""Foldability evaluation does not require a dataset split.

Provides utilities to deduplicate generated sequences and to bundle
the reference pool with its pre-computed novelty set / AA distribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from .metrics import precompute_reference

logger = logging.getLogger(__name__)


def dedupe_generated(sequences: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for seq in sequences:
        if seq not in seen:
            seen.add(seq)
            out.append(seq)
    return out


@dataclass
class ReferencePool:
    """Reference corpus plus the derived structures the evaluator needs."""

    sequences: List[str]
    reference_set: Set[str] = field(default_factory=set)
    aa_distribution: Dict[str, float] = field(default_factory=dict)


def prepare_reference_pool(
    sequences: List[str],
    max_ref_for_aa: Optional[int] = None,
    seed: int = 42,
) -> ReferencePool:
    """Pre-compute the membership set and AA distribution.

    Both are reused across the (point-estimate) metric pass and the
    bootstrap CI loop, so this avoids re-iterating ≈ 1 M reference
    sequences per resample.
    """
    logger.info(
        "Pre-computing reference set + AA distribution from %d sequences "
        "(max_ref_for_aa=%s)",
        len(sequences),
        max_ref_for_aa,
    )
    ref_set, aa_dist = precompute_reference(
        sequences, max_ref_for_set=max_ref_for_aa, seed=seed
    )
    return ReferencePool(
        sequences=list(sequences), reference_set=ref_set, aa_distribution=aa_dist
    )
