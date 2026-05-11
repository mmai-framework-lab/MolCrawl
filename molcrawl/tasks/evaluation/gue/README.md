# GUE evaluation

Phase 3 genome classification benchmark (28 sub-tasks).  Follows the
DNABERT-2 distribution: each sub-task ships `train.csv`, `dev.csv`, and
`test.csv` with a `sequence` / `label` schema.  The evaluator embeds
through `ModelAdapter.embed`, fits a logistic-regression probe, and
reports accuracy, macro F1, and MCC (binary) / none (multi-class).

## Entry point

```bash
bash workflows/eval-gue.sh
```
