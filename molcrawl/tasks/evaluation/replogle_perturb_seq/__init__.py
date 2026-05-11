"""Replogle Perturb-seq perturbation response evaluator.

Predicts the gene-expression delta caused by a gene knockdown and
compares against the experimental measurement via Spearman / Pearson
correlation (per perturbation, aggregated as the mean).
"""

from .evaluator import ReploglePerturbSeqEvaluator

__all__ = ["ReploglePerturbSeqEvaluator"]
