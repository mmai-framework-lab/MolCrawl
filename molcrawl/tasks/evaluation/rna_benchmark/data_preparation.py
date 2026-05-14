"""JSONL loader for the rna_benchmark evaluator.

Each line is expected to carry ``dataset`` (group tag) and ``tokens``
(list of int ids in the model's vocabulary).  Optional ``label`` /
``token_count`` fields are preserved.

Tokens are passed through to the adapter unchanged. The HfMlm adapter
accepts pre-tokenised int lists alongside strings, so no tokenizer
round-trip is required from the evaluator side. (The legacy
``tokens_to_strings`` helper is preserved for callers that need a
textual representation, e.g. predictions log previews.)
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class CellGroup:
    name: str
    tokens: List[List[int]] = field(default_factory=list)
    labels: List[object] = field(default_factory=list)
    token_counts: List[int] = field(default_factory=list)


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
            group.tokens.append([int(t) for t in record["tokens"]])
            if "label" in record:
                group.labels.append(record["label"])
            group.token_counts.append(
                int(record.get("token_count", len(record["tokens"])))
            )
    logger.info("Loaded %d RNA benchmark groups from %s", len(groups), jsonl_path)
    return groups


def subsample_groups(
    groups: Dict[str, CellGroup],
    cells_per_group: int,
    seed: int = 42,
) -> Dict[str, CellGroup]:
    """Reproducibly down-sample each group to ``cells_per_group`` cells.

    Uses :func:`random.Random.sample` (no replacement) per group so each
    sub-sampled population is distinct. Returns a fresh dict; the input
    is not mutated.
    """
    rng = random.Random(seed)
    out: Dict[str, CellGroup] = {}
    for name, group in groups.items():
        n = len(group.tokens)
        if cells_per_group >= n:
            out[name] = group
            continue
        indices = rng.sample(range(n), cells_per_group)
        out[name] = CellGroup(
            name=group.name,
            tokens=[group.tokens[i] for i in indices],
            labels=[group.labels[i] for i in indices] if group.labels else [],
            token_counts=[group.token_counts[i] for i in indices]
            if group.token_counts
            else [],
        )
    logger.info(
        "Subsampled groups to %d cells/group (seed=%d)", cells_per_group, seed
    )
    return out


def tokens_to_strings(tokens: Sequence[Sequence[int]]) -> List[str]:
    """Render each cell as a whitespace-separated token-id string.

    Kept for backwards compatibility / debugging previews; the live
    score path now passes int lists straight to the adapter.
    """
    return [" ".join(str(int(t)) for t in cell) for cell in tokens]
