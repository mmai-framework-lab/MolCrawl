"""Tabula Sapiens JSONL loader.

Expected schema: ``tokens`` (list of gene-token ids) and ``cell_type``
(string).  Optional ``tissue`` is passed through for cross-tissue
analyses but not used by the metric pack.
"""

from __future__ import annotations

import json
import logging
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
            tokens.append(record["tokens"])
            cell_types.append(str(record["cell_type"]))
            tissues.append(str(record.get("tissue", "")))
            if max_cells is not None and len(tokens) >= int(max_cells):
                break
    logger.info("Loaded %d Tabula Sapiens cells", len(tokens))
    return {"tokens": tokens, "cell_type": cell_types, "tissue": tissues}
