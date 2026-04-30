"""MoleculeNet evaluation task (Phase 1).

Covers the 9 standard classification tasks (BBBP, Tox21, ToxCast, SIDER,
ClinTox, BACE, HIV, MUV) and the 4 regression tasks (ESOL, FreeSolv,
Lipophilicity, and a QM9 subset) with scaffold split fixtures.
"""

from .evaluator import MoleculeNetEvaluator
from .splits import TASKS, scaffold_split

__all__ = ["MoleculeNetEvaluator", "TASKS", "scaffold_split"]
