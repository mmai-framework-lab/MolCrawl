"""Concrete model adapters for evaluation tasks.

Importing this package registers all bundled adapters with
:mod:`molcrawl.tasks.evaluation._base.model_adapter`.  Evaluators should
``import molcrawl.tasks.evaluation._adapters`` once at start-up to make
sure the architecture tags they expect are available via
:func:`build_adapter`.
"""

from . import gpt2_adapter  # noqa: F401  (import registers the adapter)

__all__ = ["gpt2_adapter"]
