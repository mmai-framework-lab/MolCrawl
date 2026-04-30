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

These files are *not* importable Python modules. They are configuration
DSLs read at runtime via ``exec(open(path).read())`` from each model's
``configurator.py``, so they intentionally bind tokenizers, dataset
paths, and meta_vocab_size at top level (every name becomes a global
that the host script then picks up). Importing one directly will run
those side effects against whatever ``LEARNING_SOURCE_DIR`` happens to
be set, and the file may also load tokenizer files relative to ``cwd``.

Exclude the modality subpackages from pdoc since they are not part of
the public API surface of ``molcrawl``.
"""

__pdoc__ = {
    "compounds": False,
    "genome_sequence": False,
    "molecule_nat_lang": False,
    "protein_sequence": False,
    "rna": False,
}
