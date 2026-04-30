"""Reference corpus loading for the foldability task.

The reference corpus is typically a small curated FASTA (UniRef50
representatives at a capped length) shipped with the benchmark.  This
module only needs the sequences to compute novelty / composition
baselines; fetch + SHA256 verification lives in
``workflows/eval-protein-foldability.sh``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def load_fasta_sequences(path: Path) -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    sequences: List[str] = []
    current: List[str] = []
    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
                continue
            current.append(line)
    if current:
        sequences.append("".join(current))
    logger.info("Loaded %d reference sequences from %s", len(sequences), file_path)
    return sequences
