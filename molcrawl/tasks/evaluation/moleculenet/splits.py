"""Scaffold split fixture generation for MoleculeNet.

Implements the Bemis-Murcko scaffold split used by the MoleculeNet
benchmark, plus a plain random split as a fall-back comparison point.

Both splits are deterministic given ``seed``; the scaffold split falls
back to random when RDKit is not available, so tests can still exercise
the evaluator skeleton in environments without chemistry tooling.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Convenience alias so downstream code can ``from .splits import TASKS``.
TASKS: Tuple[str, ...] = (
    "bbbp",
    "tox21",
    "toxcast",
    "sider",
    "clintox",
    "bace",
    "hiv",
    "muv",
    "esol",
    "freesolv",
    "lipophilicity",
    "qm9_subset",
)


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _bemis_murcko_scaffold(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or ""


def _deterministic_hash(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def scaffold_split(
    smiles: Sequence[str],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> SplitResult:
    """Group molecules by Bemis-Murcko scaffold, then carve out splits.

    Larger scaffolds enter the train set first (MoleculeNet convention).
    Molecules where the scaffold cannot be computed are placed in their
    own singleton groups so the split stays deterministic.
    """
    if not 0 < val_frac + test_frac < 1:
        raise ValueError("val_frac + test_frac must be in (0, 1)")

    scaffolds: Dict[str, List[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles):
        scaffold = _bemis_murcko_scaffold(smi)
        if scaffold is None:
            scaffold = f"__unk_{_deterministic_hash(smi)}"
        scaffolds[scaffold].append(idx)

    buckets = sorted(scaffolds.values(), key=lambda b: (-len(b), b[0]))

    n = len(smiles)
    val_cutoff = n - int(n * (val_frac + test_frac))
    test_cutoff = n - int(n * test_frac)

    train: List[int] = []
    val: List[int] = []
    test: List[int] = []

    for bucket in buckets:
        if len(train) + len(bucket) <= val_cutoff:
            train.extend(bucket)
        elif len(train) + len(val) + len(bucket) <= test_cutoff:
            val.extend(bucket)
        else:
            test.extend(bucket)

    rng = np.random.default_rng(seed)
    for subset in (train, val, test):
        rng.shuffle(subset)

    logger.info(
        "Scaffold split: train=%d val=%d test=%d", len(train), len(val), len(test)
    )
    return SplitResult(
        train_idx=np.asarray(train, dtype=int),
        val_idx=np.asarray(val, dtype=int),
        test_idx=np.asarray(test, dtype=int),
    )


def random_split(
    n: int,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> SplitResult:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    val_size = int(n * val_frac)
    test_size = int(n * test_frac)
    train_size = n - val_size - test_size
    return SplitResult(
        train_idx=indices[:train_size],
        val_idx=indices[train_size : train_size + val_size],
        test_idx=indices[train_size + val_size :],
    )


def apply_split(df: pd.DataFrame, split: SplitResult):
    return (
        df.iloc[split.train_idx].reset_index(drop=True),
        df.iloc[split.val_idx].reset_index(drop=True),
        df.iloc[split.test_idx].reset_index(drop=True),
    )
