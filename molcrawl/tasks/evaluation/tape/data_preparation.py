"""TAPE JSONL loader + sampling helpers.

The upstream release ships per-task JSONL files with ``primary`` (amino
acid sequence) plus a task-specific label column. We follow that
convention (see :mod:`prepare_csv` for how the mirrors are normalised
into this schema) and keep the label-column name configurable per task.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TAPETaskSpec:
    name: str
    task_type: str  # "classification" | "regression" | "sequence_labeling"
    sequence_column: str = "primary"
    label_column: str = "log_fluorescence"
    num_classes: int = 2
    metadata: Dict[str, str] = field(default_factory=dict)


TASKS: Sequence[str] = (
    "secondary_structure_3",
    "secondary_structure_8",
    "contact_prediction",
    "remote_homology",
    "fluorescence",
    "stability",
)


def get_spec(name: str) -> TAPETaskSpec:
    if name == "secondary_structure_3":
        return TAPETaskSpec(name, "sequence_labeling", label_column="ss3", num_classes=3)
    if name == "secondary_structure_8":
        return TAPETaskSpec(name, "sequence_labeling", label_column="ss8", num_classes=8)
    if name == "contact_prediction":
        return TAPETaskSpec(name, "sequence_labeling", label_column="tertiary", num_classes=2)
    if name == "remote_homology":
        return TAPETaskSpec(name, "classification", label_column="fold_label", num_classes=1195)
    if name == "fluorescence":
        return TAPETaskSpec(name, "regression", label_column="log_fluorescence")
    if name == "stability":
        return TAPETaskSpec(name, "regression", label_column="stability_score")
    raise KeyError(f"Unknown TAPE task: {name}")


def load_jsonl(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    records: List[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    logger.info("Loaded %d TAPE records from %s", len(records), path)
    return records


def to_frame(records: Iterable[dict], spec: TAPETaskSpec) -> pd.DataFrame:
    df = pd.DataFrame(list(records))
    missing = [c for c in (spec.sequence_column, spec.label_column) if c not in df.columns]
    if missing:
        raise ValueError(
            f"TAPE records missing columns {missing}. Available: {list(df.columns)}"
        )
    return df


def stratified_subsample(
    df: pd.DataFrame,
    n_examples: int,
    spec: TAPETaskSpec,
    seed: int = 42,
) -> pd.DataFrame:
    """Reproducibly down-sample a TAPE split.

    * Classification: class-balanced if there are ≥ 2 distinct labels with
      enough rows, else uniform random.
    * Regression: quantile-binned by the label column so the perplexity
      / Spearman number reflects the corpus tail.
    * Sequence labelling: uniform random (per-residue label balancing is
      infeasible at row level).
    """
    if n_examples >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    label_col = spec.label_column

    if spec.task_type == "classification" and label_col in df.columns:
        pool = df.dropna(subset=[label_col])
        # remote_homology has 1195 fold classes; class-balanced sampling
        # past a few hundred examples mostly degenerates to "1 row per class".
        # Cap quotas at 4 per class so we still see diverse folds without
        # blowing up.
        classes = sorted(pool[label_col].dropna().unique())
        if len(classes) >= 2 and len(classes) * 4 >= n_examples:
            # n_examples small enough that 4-per-class is achievable
            quota_per_class = max(1, n_examples // len(classes))
            parts = []
            for cls in rng.choice(classes, size=min(n_examples, len(classes)), replace=False):
                sub = pool[pool[label_col] == cls]
                take = min(quota_per_class, len(sub))
                state = int(rng.integers(0, 2**32 - 1))
                parts.append(sub.sample(n=take, random_state=state))
            sampled = pd.concat(parts, ignore_index=False)
            if len(sampled) > n_examples:
                sampled = sampled.sample(n=n_examples, random_state=seed)
        elif len(classes) >= 2:
            parts = []
            base = n_examples // len(classes)
            remainder = n_examples - base * len(classes)
            for i, cls in enumerate(classes):
                quota = base + (1 if i < remainder else 0)
                sub = pool[pool[label_col] == cls]
                take = min(quota, len(sub))
                state = int(rng.integers(0, 2**32 - 1))
                parts.append(sub.sample(n=take, random_state=state))
            sampled = pd.concat(parts, ignore_index=False)
        else:
            state = int(rng.integers(0, 2**32 - 1))
            sampled = df.sample(n=n_examples, random_state=state)
        logger.info(
            "stratified_subsample (classification, %s): %d rows from %d candidates",
            label_col,
            len(sampled),
            len(df),
        )
        return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    if spec.task_type == "regression" and label_col in df.columns:
        pool = df.dropna(subset=[label_col])
        if len(pool) > 10:
            n_bins = min(10, max(2, n_examples // 25))
            try:
                bin_ids = pd.qcut(
                    pool[label_col].astype(float),
                    q=n_bins,
                    labels=False,
                    duplicates="drop",
                ).to_numpy()
            except Exception:
                bin_ids = np.zeros(len(pool), dtype=int)
            parts = []
            unique_bins = sorted(np.unique(bin_ids[~pd.isna(bin_ids)]))
            base = n_examples // max(1, len(unique_bins))
            remainder = n_examples - base * len(unique_bins)
            for i, b in enumerate(unique_bins):
                quota = base + (1 if i < remainder else 0)
                sub = pool[bin_ids == b]
                take = min(quota, len(sub))
                state = int(rng.integers(0, 2**32 - 1))
                parts.append(sub.sample(n=take, random_state=state))
            sampled = pd.concat(parts, ignore_index=False)
            logger.info(
                "stratified_subsample (regression, %s): %d rows across %d quantile bins",
                label_col,
                len(sampled),
                len(unique_bins),
            )
            return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    state = int(rng.integers(0, 2**32 - 1))
    return df.sample(n=n_examples, random_state=state).reset_index(drop=True)
