"""COSMIC sampling helpers.

Reuses ClinVar's chromosome split fixture and adds :func:`sample_cosmic`,
a class-balanced reproducible sampler that optionally stratifies by
``MUTATION_SIGNIFICANCE_TIER`` so tier-1 (high-confidence driver) and
tier-2 rows are not under-represented when the input pool is dominated
by tier-3 passenger calls.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation.clinvar.splits import chromosome_split

logger = logging.getLogger(__name__)

__all__ = ["chromosome_split", "sample_cosmic"]


def sample_cosmic(
    df: pd.DataFrame,
    n_per_class: Optional[int] = None,
    stratify_tier: bool = True,
    seed: int = 42,
    label_column: str = "cosmic_label",
    tier_column: str = "MUTATION_SIGNIFICANCE_TIER",
) -> pd.DataFrame:
    """Return a reproducible, class-balanced COSMIC sample.

    Parameters
    ----------
    df:
        Labelled cosmic dataframe from :func:`load_cosmic` (must include
        ``reference_sequence`` / ``variant_sequence`` / ``cosmic_label``).
    n_per_class:
        Target rows per class (0 = neutral / 1 = pathogenic). ``None``
        returns the dataframe unchanged.
    stratify_tier:
        When ``True`` (default), draw within each class proportionally
        across ``MUTATION_SIGNIFICANCE_TIER`` values present in the pool.
        Tiers 1 / 2 (driver) and tier 3 (passenger) are otherwise wildly
        unbalanced — without stratification a 100-row pathogenic draw
        would essentially never see a tier-2 row.
    seed:
        Random seed.
    label_column:
        Binary label column (1 = pathogenic, 0 = neutral).
    tier_column:
        Significance-tier column emitted by :mod:`prepare_csv`.
    """
    if n_per_class is None:
        logger.info(
            "sample_cosmic: n_per_class=None → returning full dataframe (%d rows)",
            len(df),
        )
        return df.reset_index(drop=True)

    if label_column not in df.columns:
        raise ValueError(
            f"sample_cosmic requires a {label_column!r} column; got {list(df.columns)}"
        )

    rng = np.random.default_rng(seed)
    parts = []
    for label in (0, 1):
        pool = df[df[label_column] == label]
        if len(pool) == 0:
            raise ValueError(
                f"No rows available for {label_column}={label}; cannot sample. "
                "Re-run prepare_csv with --per-class large enough to populate "
                "both classes."
            )
        if len(pool) < n_per_class:
            logger.warning(
                "sample_cosmic: only %d rows for %s=%d, requested %d — "
                "drawing all available.",
                len(pool),
                label_column,
                label,
                n_per_class,
            )
            sampled = pool.copy()
        elif stratify_tier and tier_column in pool.columns and pool[tier_column].nunique() > 1:
            sampled = _stratified_tier_sample(pool, n_per_class, tier_column, rng)
        else:
            state = int(rng.integers(0, 2**32 - 1))
            sampled = pool.sample(n=n_per_class, random_state=state)
        parts.append(sampled)
        _log_class_composition(label, sampled, tier_column)

    out = (
        pd.concat(parts, ignore_index=False)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    logger.info(
        "sample_cosmic: produced %d rows (n_per_class=%d, seed=%d, stratify_tier=%s)",
        len(out),
        n_per_class,
        seed,
        stratify_tier,
    )
    return out


def _stratified_tier_sample(
    pool: pd.DataFrame,
    n_per_class: int,
    tier_column: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    tiers = sorted(pool[tier_column].astype(str).unique())
    base = n_per_class // len(tiers)
    remainder = n_per_class - base * len(tiers)

    parts = []
    indices_taken = []
    for i, tier in enumerate(tiers):
        quota = base + (1 if i < remainder else 0)
        sub = pool[pool[tier_column].astype(str) == tier]
        take = min(quota, len(sub))
        if take <= 0:
            continue
        state = int(rng.integers(0, 2**32 - 1))
        drawn = sub.sample(n=take, random_state=state)
        parts.append(drawn)
        indices_taken.extend(drawn.index.tolist())

    sampled = pd.concat(parts) if parts else pool.iloc[0:0]
    shortfall = n_per_class - len(sampled)
    if shortfall > 0:
        leftover = pool.drop(index=indices_taken)
        if len(leftover) >= shortfall:
            state = int(rng.integers(0, 2**32 - 1))
            sampled = pd.concat(
                [sampled, leftover.sample(n=shortfall, random_state=state)]
            )
    return sampled


def _log_class_composition(label: int, sampled: pd.DataFrame, tier_column: str) -> None:
    name = "pathogenic" if label == 1 else "neutral"
    if tier_column in sampled.columns:
        by_tier = (
            sampled[tier_column]
            .astype(str)
            .value_counts()
            .sort_index()
            .to_dict()
        )
        logger.info(
            "sample_cosmic: %s=%d rows across tiers %s",
            name,
            len(sampled),
            by_tier,
        )
    else:
        logger.info("sample_cosmic: %s=%d rows", name, len(sampled))
