# MOSES generation-quality evaluation

Phase 1 generation benchmark for compound decoders (GPT-2 compound at
multiple sizes).  Follows the MOSES / GuacaMol distribution-learning
protocol: validity, uniqueness, novelty, internal diversity, plus the
optional FCD / SNN / Fragment / Scaffold metrics from the reference
`moses` package when it is installed.

## Input layout

```
LEARNING_SOURCE_DIR/eval/moses/
    train.csv            # reference training SMILES (ZINC-derived)
    test.csv
    test_scaffolds.csv
    manifest.json        # source URL, SHA-256, license, fetch date
```

## Metrics

- `validity`, `uniqueness`, `novelty`, `internal_diversity` from
  `_base.metric_registry`.
- Best-effort `moses.*` entries via `metrics.optional_extended_metrics`
  (FCD, SNN, Fragment, Scaffold, IntDiv1/2, logP / QED / weight / SA
  distribution Frechet distances) when the `moses` package is
  importable.

## Entry point

```bash
bash workflows/eval-moses.sh
```
