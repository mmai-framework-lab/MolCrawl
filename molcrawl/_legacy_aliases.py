"""Backward-compatibility import aliases for the package-layout refactor.

Registers a :class:`sys.meta_path` finder so legacy dotted paths resolve to
the same module objects as their canonical counterparts, e.g.

  ``molcrawl.compounds.dataset.prepare_gpt2``
      -> ``molcrawl.data.compounds.dataset.prepare_gpt2``

Using a finder (instead of a plain ``sys.modules`` swap in each shim's
``__init__``) is important: it guarantees that deeply-nested submodule
imports via the legacy path resolve to the *same* module object as the
canonical import, avoiding duplicated module state.

This module is intended to be imported exactly once from
:mod:`molcrawl.__init__`. It will be removed in stage 4 of the
package-layout refactor, once all callers migrate to canonical paths.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from typing import Optional

_ALIASES = {
    "molcrawl.compounds": "molcrawl.data.compounds",
    "molcrawl.protein_sequence": "molcrawl.data.protein_sequence",
    "molcrawl.genome_sequence": "molcrawl.data.genome_sequence",
    "molcrawl.rna": "molcrawl.data.rna",
    "molcrawl.molecule_nat_lang": "molcrawl.data.molecule_nat_lang",
    "molcrawl.gpt2": "molcrawl.models.gpt2",
    "molcrawl.bert": "molcrawl.models.bert",
    "molcrawl.chemberta2": "molcrawl.models.chemberta2",
    "molcrawl.esm2": "molcrawl.models.esm2",
    "molcrawl.dnabert2": "molcrawl.models.dnabert2",
    "molcrawl.rnaformer": "molcrawl.models.rnaformer",
}


def _translate(name: str) -> Optional[str]:
    for legacy, canonical in _ALIASES.items():
        if name == legacy:
            return canonical
        if name.startswith(legacy + "."):
            return canonical + name[len(legacy):]
    return None


class _AliasLoader(importlib.abc.Loader):
    """Loader that returns an already-loaded canonical module."""

    def __init__(self, canonical_module):
        self._canonical_module = canonical_module

    def create_module(self, spec):
        return self._canonical_module

    def exec_module(self, module):
        pass


class _LegacyAliasFinder(importlib.abc.MetaPathFinder):
    """Route legacy ``molcrawl.<modality>.*`` imports to ``molcrawl.data.<modality>.*``."""

    def find_spec(self, fullname, path, target=None):
        canonical_name = _translate(fullname)
        if canonical_name is None:
            return None
        canonical_module = importlib.import_module(canonical_name)
        loader = _AliasLoader(canonical_module)
        spec = importlib.machinery.ModuleSpec(fullname, loader)
        spec.submodule_search_locations = getattr(canonical_module, "__path__", None)
        return spec


def install() -> None:
    if not any(isinstance(f, _LegacyAliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _LegacyAliasFinder())
