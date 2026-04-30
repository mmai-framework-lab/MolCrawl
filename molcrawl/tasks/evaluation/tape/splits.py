"""TAPE uses fixed per-task splits; this helper loads them.

Expected layout (as per upstream release)::

    <task_dir>/
        <task>_train.json
        <task>_valid.json
        <task>_test.json   # (or _casp12 / _cb513 for structure)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from .data_preparation import load_jsonl


def load_splits(
    task_dir: Path,
    task_name: str,
    split_names: Iterable[str] = ("train", "valid", "test"),
) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for split in split_names:
        path = Path(task_dir) / f"{task_name}_{split}.json"
        if not path.exists():
            continue
        out[split] = load_jsonl(path)
    if not out:
        raise FileNotFoundError(
            f"No TAPE split JSONL files found under {task_dir} for {task_name}"
        )
    return out
