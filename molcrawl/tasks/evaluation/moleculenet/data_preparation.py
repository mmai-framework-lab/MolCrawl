"""MoleculeNet data loading with manifest support.

The original MoleculeNet CSVs live under
``LEARNING_SOURCE_DIR/eval/moleculenet/<task>/``.  Each task directory is
expected to contain

* ``raw.csv`` (columns: ``smiles`` and one-or-more label columns)
* ``manifest.json`` (source URL, SHA-256, license, fetch date)

The fetching script is intentionally not committed here: large data
downloads belong in ``workflows/eval-moleculenet.sh``.  This module just
normalises the CSV and canonicalises SMILES so the evaluator and the
scaffold splitter get a stable input.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MoleculeNetTaskSpec:
    """Description of a MoleculeNet sub-task."""

    name: str
    smiles_column: str
    label_columns: Sequence[str]
    task_type: str  # "classification" | "regression"
    description: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


def canonicalise_smiles(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem
    except ImportError:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_manifest(task_dir: Path) -> Dict[str, str]:
    manifest_path = Path(task_dir) / "manifest.json"
    if not manifest_path.exists():
        logger.warning("manifest.json missing for %s", task_dir)
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_dataset(
    task_dir: Path,
    spec: MoleculeNetTaskSpec,
    filename: str = "raw.csv",
) -> pd.DataFrame:
    """Load and canonicalise a MoleculeNet task CSV."""
    csv_path = Path(task_dir) / filename
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    missing = [c for c in [spec.smiles_column, *spec.label_columns] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns missing from {csv_path}: {missing}. Available: {list(df.columns)}"
        )

    df = df[[spec.smiles_column, *spec.label_columns]].copy()
    df[spec.smiles_column] = df[spec.smiles_column].map(canonicalise_smiles)
    df = df.dropna(subset=[spec.smiles_column]).reset_index(drop=True)
    logger.info("MoleculeNet %s: loaded %d examples", spec.name, len(df))
    return df


# ---------------------------------------------------------------------------
# Built-in registry of the 13 standard tasks
# ---------------------------------------------------------------------------

_CLASSIFICATION_TASKS: List[MoleculeNetTaskSpec] = [
    MoleculeNetTaskSpec("bbbp", "smiles", ["p_np"], "classification"),
    MoleculeNetTaskSpec(
        "tox21",
        "smiles",
        [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
            "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE",
            "SR-MMP", "SR-p53",
        ],
        "classification",
    ),
    MoleculeNetTaskSpec("toxcast", "smiles", ["TOX_CAST_ANY"], "classification",
                        description="Aggregated binary label; raw file has >600 assays"),
    MoleculeNetTaskSpec("sider", "smiles", ["SIDER_ANY"], "classification"),
    MoleculeNetTaskSpec("clintox", "smiles", ["FDA_APPROVED", "CT_TOX"], "classification"),
    MoleculeNetTaskSpec("bace", "mol", ["Class"], "classification"),
    MoleculeNetTaskSpec("hiv", "smiles", ["HIV_active"], "classification"),
    MoleculeNetTaskSpec("muv", "smiles", ["MUV_ANY"], "classification"),
]

_REGRESSION_TASKS: List[MoleculeNetTaskSpec] = [
    MoleculeNetTaskSpec("esol", "smiles", ["measured log solubility in mols per litre"], "regression"),
    MoleculeNetTaskSpec("freesolv", "smiles", ["expt"], "regression"),
    MoleculeNetTaskSpec("lipophilicity", "smiles", ["exp"], "regression"),
    MoleculeNetTaskSpec("qm9_subset", "smiles", ["mu"], "regression",
                        description="QM9 dipole moment as representative regression target"),
]


def default_tasks() -> List[MoleculeNetTaskSpec]:
    """Return the 13 standard MoleculeNet tasks."""
    return [*_CLASSIFICATION_TASKS, *_REGRESSION_TASKS]


def get_task(name: str) -> MoleculeNetTaskSpec:
    for spec in default_tasks():
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown MoleculeNet task: {name}")
