"""Per-modality pre-training configs.

Layout: ``<modality>/<arch>[_<variant>][_<size>].py``

- ``<modality>`` is one of ``compounds``, ``protein_sequence``,
  ``genome_sequence``, ``rna``, ``molecule_nat_lang``.
- ``<arch>`` is one of ``gpt2``, ``bert``, ``chemberta2``, ``esm2``,
  ``dnabert2``, ``rnaformer``.
- ``<variant>`` is an optional dataset or benchmark tag
  (``chembl``, ``guacamol``, ``clinvar``, ``proteingym``, ``celltype``,
  ``mol_instructions``, ...).
- ``<size>`` is an optional explicit size suffix
  (``small`` / ``medium`` / ``large`` / ``xl``). When the model family
  takes its size as a runtime argument (e.g. BERT, ChemBERTa-2, ESM-2,
  DNABERT-2, RNAformer) a single file per variant is provided.
"""
