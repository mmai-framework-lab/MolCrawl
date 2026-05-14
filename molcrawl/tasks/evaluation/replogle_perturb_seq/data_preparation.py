"""Replogle Perturb-seq loader + sampling helpers.

Expected schema (CSV emitted by :mod:`prepare_csv`): ``perturbation``
(gene KO target), ``baseline`` (control expression vector, JSON-
serialised list), ``perturbed`` (post-KO vector). The evaluator
computes ``delta = perturbed - baseline`` and treats that as the
perturbation response signal.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_replogle(path: Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = {"perturbation", "baseline", "perturbed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Replogle file missing required columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )

    def _parse(value: object):
        if isinstance(value, str):
            return json.loads(value)
        return value

    df["baseline"] = df["baseline"].map(_parse)
    df["perturbed"] = df["perturbed"].map(_parse)
    logger.info("Loaded %d perturbations from %s", len(df), csv_path)
    return df


def stratified_subsample(
    df: pd.DataFrame,
    n_examples: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Reproducibly down-sample to ``n_examples`` rows.

    We keep the strongest KO effects (by mean |delta|) for half the
    quota — those are where any model signal is easiest to detect —
    and fill the rest with uniform random rows so the distribution
    still covers weaker effects.
    """
    if n_examples >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)

    def _strength(row) -> float:
        b = np.asarray(row["baseline"], dtype=float)
        p = np.asarray(row["perturbed"], dtype=float)
        if b.size == 0 or p.size == 0:
            return 0.0
        return float(np.mean(np.abs(p - b)))

    df = df.copy()
    df["_strength"] = df.apply(_strength, axis=1)

    n_strong = max(1, n_examples // 2)
    strong_idx = df["_strength"].sort_values(ascending=False).head(n_strong).index
    remaining = df.drop(index=strong_idx)
    n_random = n_examples - n_strong
    if n_random > 0 and len(remaining) > 0:
        random_idx = remaining.sample(
            n=min(n_random, len(remaining)),
            random_state=int(rng.integers(0, 2**32 - 1)),
        ).index
        sampled_idx = strong_idx.union(random_idx)
    else:
        sampled_idx = strong_idx

    sampled = df.loc[sampled_idx].drop(columns=["_strength"], errors="ignore")
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(
        "stratified_subsample: %d rows (%d strong + %d random)",
        len(sampled),
        n_strong,
        max(0, len(sampled) - n_strong),
    )
    return sampled
