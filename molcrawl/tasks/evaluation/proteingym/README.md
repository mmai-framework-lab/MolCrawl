# ProteinGym mutation-effect evaluation

Migration of the legacy `molcrawl/evaluation/{bert,gpt2}/proteingym_*`
scripts into the new task-centric layout.  Architecture differences
(GPT-2 protein, BERT protein, ESM-2) are absorbed by the model
adapters.

## Protocol

Zero-shot variant fitness prediction: score both wildtype and mutated
sequences through `ModelAdapter.score_likelihood`, take the difference,
and correlate against the experimental `DMS_score`.  Reports Spearman
and Pearson correlations, plus AUROC / AUPRC when a binary
`DMS_bin_score` column is present.

## Entry point

```bash
bash workflows/eval-proteingym.sh
```
