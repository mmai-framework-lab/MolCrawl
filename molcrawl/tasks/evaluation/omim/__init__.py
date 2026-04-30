"""OMIM gene-disease association evaluator.

Uses the same reference / variant likelihood machinery as ClinVar but
accepts a free-form disease category column that gets collapsed to a
binary label (known-disease vs control) for the main metric pack.
"""

from .evaluator import OMIMEvaluator

__all__ = ["OMIMEvaluator"]
