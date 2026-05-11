"""GUE (Genome Understanding Evaluation) benchmark.

Covers 28 classification tasks over promoter, enhancer, splice, TF
binding, and histone-modification datasets.  Each sub-task is a simple
supervised classification problem with its own train / dev / test CSVs.
"""

from .evaluator import GUEEvaluator
from .data_preparation import TASKS as GUE_TASKS

__all__ = ["GUEEvaluator", "GUE_TASKS"]
