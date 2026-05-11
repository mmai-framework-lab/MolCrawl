"""TAPE JSONL loader.

The upstream release ships per-task JSON files with ``primary`` (amino
acid sequence) plus a task-specific label column.  We follow the same
convention here and keep the label-column name configurable per task.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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
