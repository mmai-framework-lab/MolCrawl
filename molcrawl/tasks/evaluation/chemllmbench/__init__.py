"""ChemLLMBench evaluation.

Nine tasks covering name conversion, property QA, reaction prediction,
retro-synthesis, yield prediction, etc.  Each sub-task is a free-form
QA problem: the model reads a prompt and produces an answer, compared
against a gold answer by task-specific scoring.
"""

from .evaluator import ChemLLMBenchEvaluator
from .data_preparation import TASKS as CHEMLLMBENCH_TASKS

__all__ = ["ChemLLMBenchEvaluator", "CHEMLLMBENCH_TASKS"]
