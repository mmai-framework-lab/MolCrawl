"""Reproducible sampling + study-based grouping for ProteinGym.

Each ProteinGym CSV corresponds to a single protein / assay and
typically contains a few thousand rows with a reasonably balanced
DMS-score distribution. The main sampling need within one CSV is
reproducibility + an optional cap for smoke runs; ``stratify_bin``
leverages ``DMS_bin_score`` (when present) to draw equal numbers from
the functional / non-functional halves so AUROC / AUPRC stay
well-defined on small samples.

``group_by_study`` / ``filter_studies`` are preserved for downstream
workflows that want to aggregate per-assay metrics.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def sample_proteingym(
    df: pd.DataFrame,
    n_examples: Optional[int] = None,
    stratify_bin: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a reproducible ProteinGym sample.

    Parameters
    ----------
    df:
        Output of :func:`load_proteingym`.
    n_examples:
        Optional cap on the total returned rows. ``None`` (or a value
        ``>= len(df)``) returns the full assay unchanged.
    stratify_bin:
        When ``True`` and a ``DMS_bin_score`` column is present with
        ≥ 2 labels, draws roughly ``n_examples / 2`` rows from each
        binary class so downstream AUROC / AUPRC have both classes
        populated. Falls back to uniform within-dataframe sampling.
    seed:
        Random seed for reproducibility.
    """
    if n_examples is None or n_examples >= len(df):
        logger.info(
            "sample_proteingym: returning full dataframe (%d rows; n_examples=%s)",
            len(df),
            n_examples,
        )
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    if (
        stratify_bin
        and "DMS_bin_score" in df.columns
        and df["DMS_bin_score"].dropna().nunique() >= 2
    ):
        per_bin = n_examples // 2
        remainder = n_examples - per_bin * 2
        parts = []
        for i, label in enumerate(sorted(df["DMS_bin_score"].dropna().unique())):
            take = per_bin + (1 if i < remainder else 0)
            pool = df[df["DMS_bin_score"] == label]
            take = min(take, len(pool))
            state = int(rng.integers(0, 2**32 - 1))
            parts.append(pool.sample(n=take, random_state=state))
        sampled = pd.concat(parts, ignore_index=False)
        logger.info(
            "sample_proteingym: bin-stratified sample of %d rows (per-bin target=%d)",
            len(sampled),
            per_bin,
        )
    else:
        if stratify_bin:
            logger.info(
                "sample_proteingym: stratify_bin requested but DMS_bin_score "
                "unavailable — falling back to uniform random sampling."
            )
        state = int(rng.integers(0, 2**32 - 1))
        sampled = df.sample(n=n_examples, random_state=state)
        logger.info("sample_proteingym: random sample of %d rows", len(sampled))

    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def group_by_study(df: pd.DataFrame, column: str = "study") -> Dict[str, pd.DataFrame]:
    """Return a ``{study_name: subframe}`` mapping."""
    if column not in df.columns:
        return {"all": df}
    groups: Dict[str, pd.DataFrame] = {}
    for name, sub in df.groupby(column):
        groups[str(name)] = sub.reset_index(drop=True)
    return groups


def filter_studies(df: pd.DataFrame, studies: Iterable[str], column: str = "study") -> pd.DataFrame:
    if column not in df.columns:
        return df
    studies_set = set(studies)
    return df[df[column].isin(studies_set)].reset_index(drop=True)
