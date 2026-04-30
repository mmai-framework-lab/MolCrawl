# COSMIC evaluation

Phase 3 variant-effect benchmark.  Mirrors the ClinVar protocol:
reference vs variant likelihood, F1-optimal threshold, binary
classification metrics.  The only COSMIC-specific concern is the
label mapping exposed through ``label_column`` +
``DEFAULT_LABEL_MAP``.
