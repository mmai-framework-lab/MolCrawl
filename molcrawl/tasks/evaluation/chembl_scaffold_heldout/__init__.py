"""ChEMBL scaffold-held-out evaluation (Phase 1).

Scaffold-based held-out test set carved out of the same ChEMBL corpus
used to train the compound foundation models.  Two evaluation modes:

* decoder: held-out perplexity (arch=gpt2).
* encoder: linear probe over a known assay column
  (arch=chemberta2 / bert compound).
"""

from .evaluator import ChEMBLScaffoldHeldoutEvaluator

__all__ = ["ChEMBLScaffoldHeldoutEvaluator"]
