"""ChemLLMBench loader.

The upstream ChemLLMBench release ships one JSONL per sub-task with
``prompt`` + ``answer`` (plus optional ``metadata``) on each line.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


TASKS: Tuple[str, ...] = (
    "name_conversion",
    "property_prediction",
    "reaction_prediction",
    "retrosynthesis",
    "yield_prediction",
    "molecule_captioning",
    "text_guided_generation",
    "molecule_design",
    "smiles_understanding",
)


TASK_TYPE = {
    "name_conversion": "exact",
    "property_prediction": "exact",
    "reaction_prediction": "smiles",
    "retrosynthesis": "smiles",
    "yield_prediction": "regression",
    "molecule_captioning": "text",
    "text_guided_generation": "smiles",
    "molecule_design": "smiles",
    "smiles_understanding": "exact",
}


@dataclass
class ChemLLMBenchExample:
    prompt: str
    answer: str
    metadata: dict


def load_jsonl(path: Path) -> List[ChemLLMBenchExample]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    out: List[ChemLLMBenchExample] = []
    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            out.append(
                ChemLLMBenchExample(
                    prompt=str(record["prompt"]),
                    answer=str(record["answer"]),
                    metadata=record.get("metadata", {}),
                )
            )
    logger.info("Loaded %d ChemLLMBench examples from %s", len(out), file_path)
    return out
