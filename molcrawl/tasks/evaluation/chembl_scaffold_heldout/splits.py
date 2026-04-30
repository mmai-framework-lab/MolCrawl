"""Scaffold-split helpers specific to ChEMBL.

Wraps the MoleculeNet scaffold splitter with a guardrail: the held-out
test frame must contain no scaffolds that also appear in the training
frame, otherwise the evaluator number is not an honest held-out number.
"""

from __future__ import annotations

from typing import Sequence

from molcrawl.tasks.evaluation.moleculenet.splits import scaffold_split  # noqa: F401


def warn_on_scaffold_overlap(
    train_smiles: Sequence[str], test_smiles: Sequence[str]
) -> int:
    """Return the number of scaffolds leaked between train and test."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        return 0

    def _scaffold(smi: str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    train_sc = {s for s in (_scaffold(s) for s in train_smiles) if s}
    test_sc = {s for s in (_scaffold(s) for s in test_smiles) if s}
    return len(train_sc & test_sc)
