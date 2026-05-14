"""Concrete model adapters for evaluation tasks.

Importing this package registers all bundled adapters with
:mod:`molcrawl.tasks.evaluation._base.model_adapter`.  Evaluators should
``import molcrawl.tasks.evaluation._adapters`` once at start-up to make
sure the architecture tags they expect are available via
:func:`build_adapter`.
"""

from . import gpt2_adapter  # noqa: F401  (import registers the adapter)
from . import hf_mlm_adapter  # noqa: F401  (import registers bert/esm2/chemberta2/dnabert2/rnaformer)

__all__ = ["gpt2_adapter", "hf_mlm_adapter"]
