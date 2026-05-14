"""Reference-set loading for MOSES.

The MOSES benchmark distributes three canonical CSVs: ``train.csv``,
``test.csv``, and ``test_scaffolds.csv`` (ZINC-derived). The evaluator
needs the training SMILES (for novelty) and benefits from the test /
test_scaffolds sets as a *cross-novelty* signal: a model that
"memorises" only the train side may still score high vs train, so we
also report novelty against the held-out splits.

``manifest.json`` next to these files records provenance and SHA-256.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


def load_reference_smiles(
    path: Path, column: str = "SMILES", limit: Optional[int] = None
) -> List[str]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if column not in df.columns:
        if "smiles" in df.columns:
            column = "smiles"
        else:
            raise ValueError(
                f"Reference CSV missing SMILES column (looked for {column!r}); "
                f"available: {list(df.columns)}"
            )
    smiles = df[column].astype(str).tolist()
    if limit is not None:
        smiles = smiles[: int(limit)]
    logger.info("Loaded %d reference SMILES from %s", len(smiles), csv_path)
    return smiles


def canonicalise_set(
    smiles_list: Iterable[str],
) -> Set[str]:
    """Return the set of canonical SMILES (RDKit) for fast novelty lookup.

    Falls back to a set of raw strings when RDKit is unavailable, so the
    evaluator still runs in a stripped-down environment (with the caveat
    that novelty becomes a string-equality check rather than a chemical
    one).
    """
    try:
        from rdkit import Chem, RDLogger
    except ImportError:
        logger.info(
            "RDKit unavailable; canonicalise_set falls back to raw string equality"
        )
        return {str(s) for s in smiles_list}

    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
    out: Set[str] = set()
    for s in smiles_list:
        if not isinstance(s, str) or not s:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            out.add(Chem.MolToSmiles(mol, canonical=True))
        except Exception:  # noqa: BLE001 — RDKit can raise on edge-case mols
            continue
    return out
