"""MOSES generation-quality evaluation (Phase 1).

Implements the distribution-learning evaluation pipeline used by the
MOSES benchmark (``Validity``, ``Uniqueness``, ``Novelty``, ``Internal
Diversity``), with an optional hook into the reference MOSES / GuacaMol
packages for FCD / SNN / Fragment / Scaffold metrics when those
dependencies are available.
"""

from .evaluator import MOSESEvaluator

__all__ = ["MOSESEvaluator"]
