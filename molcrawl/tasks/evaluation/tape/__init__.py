"""TAPE (Tasks Assessing Protein Embeddings) evaluation.

Covers the 5 benchmark tasks: secondary structure (SS3/SS8), contact
prediction, remote homology, fluorescence, and stability.  The evaluator
leans on embedding probes for encoder architectures (BERT protein,
ESM-2) and likelihood scoring for decoder fluorescence/stability
ablations.
"""

from .evaluator import TAPEEvaluator
from .data_preparation import TASKS as TAPE_TASKS

__all__ = ["TAPEEvaluator", "TAPE_TASKS"]
