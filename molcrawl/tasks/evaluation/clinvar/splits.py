"""Holdout splits and reproducible sampling for ClinVar.

Two helpers live here:

- :func:`chromosome_split` retains the seen / unseen split used when we
  want to compare generalisation to held-out chromosomes (e.g. chrX /
  chrY) without touching the evaluator wiring.
- :func:`sample_clinvar` produces a reproducible, class-balanced, and
  (where possible) chromosome-stratified sample for the main evaluation
  protocol. It replaces the naive ``df.head(max_examples)`` slice which
  used to bias every smoke run toward one class because the on-disk CSV
  groups rows by ``ClinicalSignificance``.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


def sample_clinvar(
    df: pd.DataFrame,
    n_per_class: Optional[int] = None,
    stratify_chrom: bool = True,
    seed: int = 42,
    chrom_column: str = "chrom",
    label_column: str = "pathogenic",
) -> pd.DataFrame:
    """Return a reproducible, class-balanced ClinVar sample.

    Parameters
    ----------
    df:
        Labelled ClinVar dataframe produced by :func:`load_clinvar`
        (must include ``reference_sequence``, ``variant_sequence``,
        ``pathogenic`` and — for stratification — a chromosome column).
    n_per_class:
        Target size per class. ``None`` returns the dataframe unchanged
        (full-run mode). When set, the same count is drawn from each
        class (the dataset's 73 % / 27 % imbalance is intentionally
        neutralised so AUROC / AUPRC are well-defined).
    stratify_chrom:
        When ``True`` (default) and ``n_per_class`` is large enough to
        allocate at least one row to each chromosome, rows are drawn
        proportionally from each chromosome within each class. This
        counteracts the per-chromosome pathogenic-rate variance
        (e.g. 86 % on chrY, 48 % on chrX, 27 % overall) so the model is
        not secretly measured on chromosome identity.
    seed:
        Random seed for reproducibility.
    chrom_column:
        Name of the chromosome column (defaults to ``"chrom"`` as
        produced by the upstream data preparation scripts).
    label_column:
        Binary label column (``1`` = pathogenic, ``0`` = benign).

    Raises
    ------
    ValueError
        If the requested ``n_per_class`` exceeds the number of rows
        available for either class.
    """

    if n_per_class is None:
        logger.info(
            "sample_clinvar: n_per_class=None → returning full dataframe (%d rows)",
            len(df),
        )
        return df.reset_index(drop=True)

    if label_column not in df.columns:
        raise ValueError(
            f"sample_clinvar requires a {label_column!r} column; got {list(df.columns)}"
        )

    rng = np.random.default_rng(seed)
    per_class_parts = []
    for label in (0, 1):
        pool = df[df[label_column] == label]
        if len(pool) < n_per_class:
            raise ValueError(
                f"Only {len(pool)} rows available for {label_column}={label}, "
                f"requested n_per_class={n_per_class}. Reduce n_per_class or "
                "disable stratified sampling."
            )

        use_chrom_stratification = (
            stratify_chrom
            and chrom_column in pool.columns
            and n_per_class >= pool[chrom_column].nunique()
        )

        if use_chrom_stratification:
            sampled = _stratified_chrom_sample(
                pool, n_per_class, chrom_column, rng
            )
        else:
            if stratify_chrom and chrom_column not in pool.columns:
                logger.info(
                    "sample_clinvar: no %r column; falling back to within-class "
                    "random sampling.",
                    chrom_column,
                )
            elif stratify_chrom:
                logger.info(
                    "sample_clinvar: n_per_class=%d < %d unique chromosomes; "
                    "falling back to within-class random sampling.",
                    n_per_class,
                    pool[chrom_column].nunique(),
                )
            state = int(rng.integers(0, 2**32 - 1))
            sampled = pool.sample(n=n_per_class, random_state=state)

        per_class_parts.append(sampled)
        _log_class_composition(label, sampled, chrom_column)

    out = (
        pd.concat(per_class_parts, ignore_index=False)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    logger.info(
        "sample_clinvar: produced %d rows (n_per_class=%d, seed=%d, stratify_chrom=%s)",
        len(out),
        n_per_class,
        seed,
        stratify_chrom,
    )
    return out


def _stratified_chrom_sample(
    pool: pd.DataFrame,
    n_per_class: int,
    chrom_column: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw ``n_per_class`` rows with a proportional chromosome quota.

    Each chromosome is allocated ``floor(n/C)`` slots, with the first
    ``n - floor(n/C)*C`` chromosomes receiving one additional slot so
    the total adds up exactly to ``n_per_class``. Any chromosome that
    cannot fill its slot yields the remainder to a global top-up pass
    from the leftover rows.
    """
    chroms = sorted(pool[chrom_column].astype(str).unique())
    base = n_per_class // len(chroms)
    remainder = n_per_class - base * len(chroms)

    parts = []
    indices_taken = []
    for i, chrom in enumerate(chroms):
        quota = base + (1 if i < remainder else 0)
        sub = pool[pool[chrom_column].astype(str) == chrom]
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
        if len(leftover) < shortfall:
            raise ValueError(
                f"Stratified top-up needs {shortfall} more rows but only "
                f"{len(leftover)} are available in this class"
            )
        state = int(rng.integers(0, 2**32 - 1))
        sampled = pd.concat(
            [sampled, leftover.sample(n=shortfall, random_state=state)]
        )

    return sampled


def _log_class_composition(
    label: int, sampled: pd.DataFrame, chrom_column: str
) -> None:
    name = "pathogenic" if label == 1 else "benign"
    if chrom_column in sampled.columns:
        by_chrom = (
            sampled[chrom_column]
            .astype(str)
            .value_counts()
            .sort_index()
            .to_dict()
        )
        logger.info(
            "sample_clinvar: %s=%d rows across %d chromosomes (%s)",
            name,
            len(sampled),
            len(by_chrom),
            by_chrom,
        )
    else:
        logger.info("sample_clinvar: %s=%d rows", name, len(sampled))
