# MoleculeNet evaluation

Phase 1 property-prediction benchmark covering the 9 classification
tasks and 4 regression tasks called out in
[docs/\_tmp/20260422-evaluator-implementation-plan.md](../../../../docs/_tmp/20260422-evaluator-implementation-plan.md).

## Input layout

```
LEARNING_SOURCE_DIR/eval/moleculenet/<task>/
    raw.csv          # smiles + label columns
    manifest.json    # source URL, SHA-256, license, fetch date
```

## Protocol

1. `data_preparation.py` canonicalises SMILES and keeps the task label
   columns named in `MoleculeNetTaskSpec.label_columns`.
2. `splits.py` produces a deterministic Bemis-Murcko scaffold split
   (80 / 10 / 10 by default).  A random-split fallback is available for
   ablations.
3. `evaluator.py::MoleculeNetEvaluator` embeds SMILES via the adapter,
   fits a linear probe on the training subset, and reports metrics on
   the test subset.  When the adapter only exposes likelihood scoring,
   the evaluator falls back to a zero-shot perplexity baseline.
4. Metrics per task:
   - classification: AUROC, AUPRC, accuracy, F1
   - regression: RMSE, MAE, R^2

## Entry point

```bash
bash workflows/eval-moleculenet.sh
```
