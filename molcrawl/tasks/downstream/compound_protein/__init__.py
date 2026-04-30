"""Compound-protein multimodal downstream task (skeleton).

This package is the landing spot for the flagship multimodal downstream
task described in
``docs/09-future_models/COMPOUND_PROTEIN_MULTIMODAL_PROCESS.ja.md``. It
will host:

- a dual encoder that aligns :class:`molcrawl.models.chemberta2` and
  :class:`molcrawl.models.esm2` embeddings on compound / protein pairs
  sourced from ChEMBL, BindingDB, PDBbind, ...
- a conditional generator that uses protein embeddings to condition the
  :class:`molcrawl.models.gpt2` compound decoder on target context
- a shared evaluation harness for binding prediction, target retrieval,
  binding affinity, novelty, and cross-family generalization

Implementation is tracked by the plan in CLAUDE.md § 実装プロセス and is
intentionally deferred to a later PR; this skeleton exists so the
``tasks/downstream`` layer has a concrete shape before the first
content lands.
"""
