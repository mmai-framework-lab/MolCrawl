# OMIM evaluation

Phase 3 gene-disease association benchmark.  Collapses the
`disease_category` column to a binary `omim_label` and then reuses the
ClinVar protocol (reference vs variant likelihood, F1-optimal threshold).
