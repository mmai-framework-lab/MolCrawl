"""Reproducible AF-stratified sampling for the gnomAD evaluator.

gnomAD allele frequencies are heavy-tailed: the vast majority of
variants sit at AF < 1e-4, so uniform random sampling effectively
picks only rare variants and gives the correlation metric almost no
common-AF leverage. :func:`sample_gnomad` bins AF on a ``log10`` scale
and draws ``n_per_bin`` rows per bin, which forces a rank-diverse test
set and lets the evaluator report per-bin Spearman to see whether the
model's likelihood signal is concentrated in a particular frequency
regime.

Variants with AF = 0 (observed but downstream-filtered alleles) are
excluded because ``log10(0)`` is undefined and they are biologically
distinct from "rare but observed".
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Re-export the chromosome_split helper from ClinVar so callers that
# import from this module keep working.
from molcrawl.tasks.evaluation.clinvar.splits import chromosome_split  # noqa: F401

logger = logging.getLogger(__name__)


DEFAULT_AF_BINS: Tuple[Tuple[float, float], ...] = (
    (0.0, 1e-5),
    (1e-5, 1e-4),
    (1e-4, 1e-3),
    (1e-3, 1e-2),
    (1e-2, 1e-1),
    (1e-1, 1.0 + 1e-12),
)


def bin_label(lo: float, hi: float) -> str:
    """Return a compact label such as ``"1e-3..1e-2"``."""
    def _fmt(x: float) -> str:
        if x == 0.0:
            return "0"
        return f"1e{int(round(math.log10(x)))}"
    return f"{_fmt(lo)}..{_fmt(hi)}"


def assign_af_bins(
    af: Union[Sequence[float], np.ndarray], bins: Sequence[Tuple[float, float]] = DEFAULT_AF_BINS
) -> np.ndarray:
    """Return a bin index per AF value; ``-1`` for values outside any bin."""
    af_arr = np.asarray(af, dtype=float)
    out = np.full(len(af_arr), -1, dtype=int)
    for idx, (lo, hi) in enumerate(bins):
        mask = (af_arr >= lo) & (af_arr < hi)
        out[mask] = idx
    return out


def sample_gnomad(
    df: pd.DataFrame,
    n_per_bin: Optional[int] = None,
    seed: int = 42,
    bins: Sequence[Tuple[float, float]] = DEFAULT_AF_BINS,
    af_column: str = "allele_frequency",
) -> pd.DataFrame:
    """Return a reproducible AF-stratified sample of gnomAD variants.

    Parameters
    ----------
    df:
        Preprocessed gnomAD table with ``reference_sequence``,
        ``variant_sequence`` and a numerical allele-frequency column.
    n_per_bin:
        Target draws per AF bin. ``None`` returns the dataframe
        unchanged (full-run mode). When set, each of the ``len(bins)``
        bins contributes at most ``n_per_bin`` rows; empty or
        undersized bins yield whatever they have (with a log line).
    seed:
        Random seed for reproducibility.
    bins:
        Sequence of ``(lo, hi)`` half-open intervals covering
        ``[0, 1]``. Defaults to six log10-spaced bins.
    af_column:
        Name of the AF column (default: ``"allele_frequency"``).
    """
    if n_per_bin is None:
        logger.info(
            "sample_gnomad: n_per_bin=None → returning full dataframe (%d rows)",
            len(df),
        )
        return df.reset_index(drop=True)

    if af_column not in df.columns:
        raise ValueError(
            f"sample_gnomad requires an AF column named {af_column!r}; "
            f"got {list(df.columns)}"
        )

    # Drop AF = 0 (filtered / unobserved alleles) — undefined in log space.
    nonzero = df[df[af_column] > 0]
    if len(nonzero) < len(df):
        logger.info(
            "sample_gnomad: dropped %d rows with AF <= 0 before binning",
            len(df) - len(nonzero),
        )
    nonzero = nonzero.reset_index(drop=True)

    bin_ids = assign_af_bins(nonzero[af_column].to_numpy(), bins)
    rng = np.random.default_rng(seed)

    parts: List[pd.DataFrame] = []
    counts_log = {}
    for i, (lo, hi) in enumerate(bins):
        mask = bin_ids == i
        pool = nonzero[mask]
        if len(pool) == 0:
            counts_log[bin_label(lo, hi)] = 0
            continue
        take = min(n_per_bin, len(pool))
        state = int(rng.integers(0, 2**32 - 1))
        drawn = pool.sample(n=take, random_state=state)
        parts.append(drawn)
        counts_log[bin_label(lo, hi)] = take
        if take < n_per_bin:
            logger.info(
                "sample_gnomad: bin %s undersized — took all %d rows (wanted %d)",
                bin_label(lo, hi),
                take,
                n_per_bin,
            )

    if not parts:
        raise ValueError(
            "sample_gnomad: no AF values fell into any of the configured bins"
        )

    out = (
        pd.concat(parts, ignore_index=False)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    logger.info(
        "sample_gnomad: produced %d rows across %d bins (seed=%d) — %s",
        len(out),
        len(parts),
        seed,
        counts_log,
    )
    return out
