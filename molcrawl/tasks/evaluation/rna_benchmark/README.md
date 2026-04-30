# rna_benchmark evaluation

Phase 4 migration of the legacy
`molcrawl/evaluation/rna/rna_benchmark_evaluation.py` script into the
task-centric layout.  Loads the tokenised-cell JSONL, groups by
`dataset`, and reports per-group mean log-likelihood + perplexity via
`ModelAdapter.score_likelihood`.
