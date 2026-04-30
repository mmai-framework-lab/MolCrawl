"""ProteinGym zero-shot mutation-effect evaluator.

Migrated from ``molcrawl.evaluation.{bert,gpt2}.proteingym_*`` into the
new task-centric layout.  The heavy-weight preparation logic in
``proteingym_data_preparation.py`` will be moved in a follow-up PR; this
package provides the arch-agnostic evaluator and a minimal loader.
"""

from .evaluator import ProteinGymEvaluator

__all__ = ["ProteinGymEvaluator"]
