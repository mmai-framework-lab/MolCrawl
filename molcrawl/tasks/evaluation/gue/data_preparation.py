"""GUE sub-task loader + sampling helpers.

Each sub-task ships three CSV files (train / dev / test) with a
``sequence`` + ``label`` schema. Task-specific metadata (binary vs
multi-class, number of classes) is kept in the :class:`GUETaskSpec`
returned by :func:`get_spec`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GUETaskSpec:
    name: str
    num_classes: int
    sequence_column: str = "sequence"
    label_column: str = "label"
    metadata: Dict[str, str] = field(default_factory=dict)


# The standard GUE release covers these 28 sub-tasks. The list mirrors
# the DNABERT-2 benchmark; binary vs multi-class follows the upstream README.
TASKS: Tuple[str, ...] = (
    "prom_300_all",
    "prom_300_notata",
    "prom_300_tata",
    "prom_core_all",
    "prom_core_notata",
    "prom_core_tata",
    "splice_reconstructed",
    "covid_variants",
    "mouse_0",
    "mouse_1",
    "mouse_2",
    "mouse_3",
    "mouse_4",
    "H3",
    "H3K14ac",
    "H3K36me3",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K79me3",
    "H3K9ac",
    "H4",
    "H4ac",
    "tf_0",
    "tf_1",
    "tf_2",
    "tf_3",
    "tf_4",
)


MULTI_CLASS_TASKS = {"splice_reconstructed": 3, "covid_variants": 9}


def get_spec(name: str) -> GUETaskSpec:
    if name not in TASKS:
        raise KeyError(f"Unknown GUE task: {name}")
    num_classes = MULTI_CLASS_TASKS.get(name, 2)
    return GUETaskSpec(name=name, num_classes=num_classes)


def load_splits(task_dir: Path) -> Dict[str, pd.DataFrame]:
    task_dir = Path(task_dir)
    out: Dict[str, pd.DataFrame] = {}
    for split_name, filename in (("train", "train.csv"), ("dev", "dev.csv"), ("test", "test.csv")):
        path = task_dir / filename
        if path.exists():
            df = pd.read_csv(path)
            if "sequence" not in df.columns or "label" not in df.columns:
                raise ValueError(
                    f"{path} missing required columns. Got {list(df.columns)}"
                )
            out[split_name] = df
    if "train" not in out:
        raise FileNotFoundError(
            f"GUE sub-task {task_dir.name} must provide train.csv"
        )
    return out


def all_task_names() -> List[str]:
    return list(TASKS)


def stratified_subsample(
    df: pd.DataFrame,
    n_examples: int,
    label_column: str = "label",
    seed: int = 42,
) -> pd.DataFrame:
    """Class-balanced subsample so each split keeps every class.

    Reproducible given ``seed``. Falls back to uniform random if the
    label column has fewer than 2 classes (already homogeneous).
    """
    if n_examples >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)

    if label_column in df.columns:
        pool = df.dropna(subset=[label_column])
        classes = sorted(pool[label_column].dropna().unique())
        if len(classes) >= 2:
            parts = []
            base = n_examples // len(classes)
            remainder = n_examples - base * len(classes)
            for i, cls in enumerate(classes):
                quota = base + (1 if i < remainder else 0)
                sub = pool[pool[label_column] == cls]
                take = min(quota, len(sub))
                state = int(rng.integers(0, 2**32 - 1))
                parts.append(sub.sample(n=take, random_state=state))
            sampled = pd.concat(parts, ignore_index=False)
            logger.info(
                "stratified_subsample (%s): %d rows across %d classes",
                label_column,
                len(sampled),
                len(classes),
            )
            return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    state = int(rng.integers(0, 2**32 - 1))
    return df.sample(n=n_examples, random_state=state).reset_index(drop=True)
