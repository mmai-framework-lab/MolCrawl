"""ChEBI-20 ships official train / val / test splits in its TSVs."""

from pathlib import Path
from typing import Dict

import pandas as pd

from .data_preparation import load_chebi20


def load_all_splits(base_dir: Path) -> Dict[str, pd.DataFrame]:
    base = Path(base_dir)
    out: Dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        for suffix in (".tsv", ".csv", ".txt"):
            candidate = base / f"{split}{suffix}"
            if candidate.exists():
                out[split] = load_chebi20(candidate)
                break
    if "test" not in out:
        raise FileNotFoundError(
            f"ChEBI-20 dir {base_dir} must contain test.tsv / .csv"
        )
    return out
