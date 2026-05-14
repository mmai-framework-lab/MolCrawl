"""Tabula Sapiens JSONL loader + sampling helpers.

Expected schema (CSV emitted by :mod:`prepare_jsonl`): ``tokens``
(list of int gene-token ids), ``cell_type`` (string), and
``tissue`` (string, optional).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def load_jsonl(path: Path, max_cells: Optional[int] = None) -> Dict[str, List]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    tokens: List[List[int]] = []
    cell_types: List[str] = []
    tissues: List[str] = []
    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tokens.append([int(t) for t in record["tokens"]])
            cell_types.append(str(record["cell_type"]))
            tissues.append(str(record.get("tissue", "")))
            if max_cells is not None and len(tokens) >= int(max_cells):
                break
    logger.info("Loaded %d Tabula Sapiens cells from %s", len(tokens), file_path)
    return {"tokens": tokens, "cell_type": cell_types, "tissue": tissues}


def stratified_subsample(
    dataset: Dict[str, List],
    n_examples: int,
    seed: int = 42,
) -> Dict[str, List]:
    """Class-balanced subsample over ``cell_type``.

    Falls back to uniform random if there are fewer than 2 classes.
    Always shuffles for downstream reproducibility.
    """
    n_total = len(dataset["tokens"])
    if n_examples >= n_total:
        return dataset

    rng = random.Random(seed)
    cell_types = dataset["cell_type"]
    classes = sorted(set(cell_types))
    idx: List[int]
    if len(classes) < 2:
        idx = rng.sample(range(n_total), n_examples)
    else:
        per_class: Dict[str, List[int]] = {c: [] for c in classes}
        for i, c in enumerate(cell_types):
            per_class[c].append(i)
        base = n_examples // len(classes)
        remainder = n_examples - base * len(classes)
        idx = []
        for i, cls in enumerate(classes):
            quota = base + (1 if i < remainder else 0)
            pool = per_class[cls]
            take = min(quota, len(pool))
            if take > 0:
                idx.extend(rng.sample(pool, take))
        rng.shuffle(idx)
        logger.info(
            "stratified_subsample: %d cells across %d cell types",
            len(idx),
            len(classes),
        )
    out: Dict[str, List] = {
        "tokens": [dataset["tokens"][i] for i in idx],
        "cell_type": [dataset["cell_type"][i] for i in idx],
        "tissue": [dataset["tissue"][i] for i in idx],
    }
    return out
