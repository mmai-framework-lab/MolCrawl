# molecule_nat_lang evaluation

Phase 5 migration of the legacy
`molcrawl/evaluation/{bert,gpt2}/molecule_nat_lang_evaluation.py`
scripts into the task-centric layout.  Scores formatted molecule /
caption pairs through `ModelAdapter.score_likelihood` and reports the
mean log-likelihood and perplexity.
