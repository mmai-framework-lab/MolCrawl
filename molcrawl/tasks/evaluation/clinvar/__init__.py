"""ClinVar pathogenicity evaluation (pilot migration from ``evaluation.gpt2``).

This package is the first task migrated to the new
``molcrawl.tasks.evaluation`` layout described in
``docs/_tmp/20260422-evaluator-implementation-plan.md``.  Architecture
differences (GPT-2 / BERT) are handled by :mod:`model_adapter`, not by
forking the evaluator.
"""

from .evaluator import ClinVarEvaluator

__all__ = ["ClinVarEvaluator"]
