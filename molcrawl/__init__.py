"""MolCrawl: unified foundation-model pipeline for molecular modalities.

The package is organized in four layers:

- :mod:`molcrawl.core` — horizontal infrastructure (paths, experiment
  tracking, shared utilities) that every other layer may depend on.
- :mod:`molcrawl.data` — per-modality data preparation, tokenizers,
  and datasets. Architecture-agnostic.
- :mod:`molcrawl.models` — model architecture implementations
  (decoder / encoder) by family. Modality-agnostic.
- :mod:`molcrawl.tasks` — training and evaluation entrypoints grouped
  by purpose: ``pretrain`` (foundation-model pre-training),
  ``evaluation`` (benchmarks by task), ``downstream`` (multimodal
  downstream tasks such as ``compound_protein``).

See :doc:`docs/10-file-tree/FILE_TREE.md` for the full layout.
"""

__version__ = "0.1.0"
