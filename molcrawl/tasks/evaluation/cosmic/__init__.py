"""COSMIC somatic-variant evaluator.

Pathogenic / benign classification of COSMIC variants using the same
reference-vs-variant likelihood protocol introduced for ClinVar.  The
only COSMIC-specific concern is the label mapping: COSMIC labels are
derived from the ``FATHMM_PREDICTION`` and / or ``MUTATION_STATUS``
columns.
"""

from .evaluator import CosmicEvaluator

__all__ = ["CosmicEvaluator"]
