"""Family-aware splits for ProteinGym.

ProteinGym ships with a stable per-study split.  When a ``study`` column
is present we simply group by that field so the evaluator can report
out-of-family Spearman correlations.
"""

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd


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
    studies = set(studies)
    return df[df[column].isin(studies)].reset_index(drop=True)
