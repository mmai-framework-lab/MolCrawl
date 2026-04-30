"""JSONL loader for the legacy rna_benchmark evaluator.

Each line is expected to carry ``dataset`` (group tag), ``tokens`` (list
of int ids), and optionally ``label``.  Tokens are preserved as-is; the
adapter is responsible for decoding them when the model needs strings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class CellGroup:
    name: str
    tokens: List[List[int]] = field(default_factory=list)
    labels: List[object] = field(default_factory=list)


def load_jsonl(path: Path, datasets: Optional[Iterable[str]] = None) -> Dict[str, CellGroup]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)
    selected: Optional[set] = {d.strip() for d in datasets} if datasets else None
    groups: Dict[str, CellGroup] = {}
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            name = record.get("dataset", "all")
            if selected is not None and name not in selected:
                continue
            group = groups.setdefault(name, CellGroup(name=name))
            group.tokens.append(record["tokens"])
            if "label" in record:
                group.labels.append(record["label"])
    logger.info("Loaded %d RNA benchmark groups", len(groups))
    return groups


def tokens_to_strings(tokens: Sequence[Sequence[int]]) -> List[str]:
    """Render each cell as a whitespace-separated token string.

    Adapters that only accept strings can tokenise once, compare on
    sub-word level, and still compute perplexity.  The representation is
    stable across adapter implementations.
    """
    return [" ".join(str(int(t)) for t in cell) for cell in tokens]
