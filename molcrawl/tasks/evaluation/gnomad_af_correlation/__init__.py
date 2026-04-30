"""gnomAD allele-frequency correlation evaluator.

Measures how strongly a genome decoder's likelihood ranks variants by
population allele frequency.  This is a generation-side metric: higher
likelihood for common variants, lower for rare ones.
"""

from .evaluator import GnomadAFEvaluator

__all__ = ["GnomadAFEvaluator"]
