"""Migration of the legacy molecule_nat_lang evaluators.

Combines the BERT + GPT-2 variants of the original
``molcrawl.evaluation.{bert,gpt2}.molecule_nat_lang_evaluation`` scripts
into one arch-agnostic evaluator.  The task measures whether the model
can score molecule / caption pairs consistently under teacher-forced
likelihood.
"""

from .evaluator import MoleculeNatLangEvaluator

__all__ = ["MoleculeNatLangEvaluator"]
