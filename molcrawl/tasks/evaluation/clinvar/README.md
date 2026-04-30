# ClinVar pathogenicity evaluation

Pilot migration for the new `molcrawl.tasks.evaluation` layout described
in [docs/\_tmp/20260422-evaluator-implementation-plan.md](../../../../docs/_tmp/20260422-evaluator-implementation-plan.md).

## Layout

- `data_preparation.py` - load a ClinVar-derived CSV/TSV/JSON and attach
  a binary `pathogenic` column.
- `splits.py` - stable chromosome-aware split (seen / unseen).
- `metrics.py` - F1-optimal threshold search and confusion-matrix pack.
- `evaluator.py` - `ClinVarEvaluator`, an arch-agnostic subclass of
  `BaseEvaluator`.
- `visualization.py` - optional ROC and histogram plots.
- `configs/` - arch/size combinations that can be driven from the CLI.

## Data

Expects a file with `reference_sequence`, `variant_sequence`, and
`ClinicalSignificance` columns.  Generation of that file is still
handled by the legacy helper
`molcrawl/evaluation/gpt2/clinvar_data_preparation.py`; migration of
that script is left for a follow-up PR.

## Running

The smoke workflow is `workflows/eval-_smoke.sh`, which calls the
entry-point below with a small sample:

```bash
python -m molcrawl.tasks.evaluation.clinvar \
    --model-path <ckpt.pt> \
    --tokenizer-path <tokenizer.model> \
    --clinvar-data <path/to/clinvar.csv> \
    --output-dir experiment_data/eval/clinvar_smoke
```
