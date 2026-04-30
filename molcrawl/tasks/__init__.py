"""Training and evaluation tasks.

Organized by purpose rather than architecture:

- :mod:`molcrawl.tasks.pretrain` — foundation model pre-training entry
  points and per-modality configs. Layout is
  ``tasks/pretrain/configs/<modality>/<arch>_<variant>.py``.
- :mod:`molcrawl.tasks.evaluation` — evaluation benchmarks grouped by
  task (ClinVar, ProteinGym, COSMIC, OMIM, ...). Architecture
  differences are absorbed at the module / CLI argument level.
- :mod:`molcrawl.tasks.downstream` — multimodal downstream tasks built
  on top of the foundation model layer (compound_protein, etc.).
"""
