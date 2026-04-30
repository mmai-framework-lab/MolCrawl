"""ChEBI-20 caption generation and SMILES generation.

Bidirectional evaluation for molecule description / instruction models.
The evaluator generates in both directions (molecule -> caption and
caption -> SMILES) and scores the outputs against reference text / SMILES.
"""

from .evaluator import ChEBI20Evaluator

__all__ = ["ChEBI20Evaluator"]
