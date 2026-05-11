"""Holdout splits for ClinVar.

The upstream evaluation protocol keeps a single test split.  This module
exposes a stable chromosome-aware split so that later work can compare
``seen`` and ``unseen`` chromosomes without touching the evaluator.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


DEFAULT_UNSEEN_CHROMOSOMES: Tuple[str, ...] = ("chr21", "chr22", "chrX", "chrY")


def chromosome_split(
    df: pd.DataFrame,
    unseen: Iterable[str] = DEFAULT_UNSEEN_CHROMOSOMES,
    column: str = "Chromosome",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into (seen, unseen) by chromosome.

    The split is a no-op (``unseen`` is empty) when the dataframe has no
    ``column`` field, so the evaluator can still run on sample data that
    lacks chromosome annotations.
    """
    if column not in df.columns:
        return df, df.iloc[0:0]
    unseen_set = {c.lower() for c in unseen}
    chrom = df[column].astype(str).str.lower()
    mask = chrom.isin(unseen_set)
    return df[~mask].copy(), df[mask].copy()
