"""Generation-side foldability evaluation for protein decoders.

Heavy structural prediction (ESMFold, AlphaFold2) is not expected to be
runnable inside every environment, so this task degrades gracefully
from pLDDT to length / composition / Pfam hit-rate proxies.
"""

from .evaluator import ProteinFoldabilityEvaluator

__all__ = ["ProteinFoldabilityEvaluator"]
