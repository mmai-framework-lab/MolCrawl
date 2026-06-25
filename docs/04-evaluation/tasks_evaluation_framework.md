# Task-centric evaluation framework

This document explains how to use the evaluation framework added under
`molcrawl/tasks/evaluation/`, which drives evaluation of the five
foundation model families through a single pipeline.  See the following
for the design and the six-phase rollout plan:

- [`docs/_tmp/20260422-evaluator-implementation-plan.md`](../_tmp/20260422-evaluator-implementation-plan.md)

## 1. Overview

All evaluation code is organised by **task**, not by architecture.
Architecture differences are absorbed by `ModelAdapter`.

```
molcrawl/tasks/evaluation/
  _base/                    # BaseEvaluator / ModelAdapter / MetricRegistry / ReportWriter
  _adapters/                # arch-specific adapters (gpt2, ...)
  _snapshot/                # cross-task rollup (Phase 6)
  <task>/
    __init__.py
    data_preparation.py
    splits.py
    metrics.py
    evaluator.py
    visualization.py
    configs/<arch>_<size>.yaml
    README.md
```

Each task ships a matching `workflows/eval-<task>.sh` driver.

## 2. Cross-cutting pieces

### 2.1 ModelHandle

A trained model is described by a `ModelHandle` that you can build
directly from CLI arguments:

```python
from molcrawl.tasks.evaluation._base import ModelHandle

handle = ModelHandle(
    arch="gpt2",                         # "gpt2" | "bert" | "chemberta2" | "esm2" | "dnabert2"
    modality="genome_sequence",          # foundation-model family
    model_path="runs_train_gpt2_genome_small/ckpt.pt",
    tokenizer_path="${LEARNING_SOURCE_DIR}/genome_sequence/spm_tokenizer.model",
    size="small",
    extras={"device": "cuda"},
)
```

See [tokenizer_paths.md](tokenizer_paths.md) for a modality / arch
matrix of which file to pass to `--tokenizer-path`.

### 2.2 Capabilities exposed by `ModelAdapter`

- `classification`
- `regression`
- `embedding`
- `likelihood`
- `generation`

Every evaluator requests only the capabilities it needs and errors out
explicitly when the adapter does not implement them.  The bundled
adapter today is `gpt2` (likelihood / generation).  Add more
architectures under `molcrawl/tasks/evaluation/_adapters/` and register
them via `register_adapter(arch, factory)`.

### 2.3 Output contract

Every task writes two files into `--output-dir`:

- `metrics.json` - machine-readable, contains `{task, modality, arch,
  category, metrics, details}`.
- `REPORT.md` - human-readable report with the 3-axis header.

The Phase 6 snapshot aggregator reads those `metrics.json` files.

## 3. Task catalogue

| Category | Task | CLI module | Workflow |
|---|---|---|---|
| variant_effect | clinvar | `molcrawl.tasks.evaluation.clinvar` | `workflows/eval-_smoke.sh` |
| variant_effect | cosmic | `molcrawl.tasks.evaluation.cosmic` | - |
| variant_effect | omim | `molcrawl.tasks.evaluation.omim` | - |
| variant_effect | proteingym | `molcrawl.tasks.evaluation.proteingym` | `workflows/eval-proteingym.sh` |
| variant_effect | gnomad_af_correlation | `molcrawl.tasks.evaluation.gnomad_af_correlation` | `workflows/eval-gnomad.sh` |
| property_prediction | moleculenet | `molcrawl.tasks.evaluation.moleculenet` | `workflows/eval-moleculenet.sh` |
| property_prediction | chembl_scaffold_heldout | `molcrawl.tasks.evaluation.chembl_scaffold_heldout` | `workflows/eval-chembl-heldout.sh` |
| property_prediction | tape | `molcrawl.tasks.evaluation.tape` | `workflows/eval-tape.sh` |
| property_prediction | deeploc | `molcrawl.tasks.evaluation.deeploc` | `workflows/eval-deeploc.sh` |
| sequence_annotation | gue | `molcrawl.tasks.evaluation.gue` | `workflows/eval-gue.sh` |
| generation_quality | moses | `molcrawl.tasks.evaluation.moses` | `workflows/eval-moses.sh` |
| foldability | protein_foldability | `molcrawl.tasks.evaluation.protein_foldability` | `workflows/eval-protein-foldability.sh` |
| cell_type_annotation | rna_benchmark | `molcrawl.tasks.evaluation.rna_benchmark` | `workflows/eval-rna-benchmark.sh` |
| cell_type_annotation | tabula_sapiens | `molcrawl.tasks.evaluation.tabula_sapiens` | `workflows/eval-tabula-sapiens.sh` |
| perturbation_response | replogle_perturb_seq | `molcrawl.tasks.evaluation.replogle_perturb_seq` | `workflows/eval-replogle-perturb-seq.sh` |
| text_alignment | molecule_nat_lang | `molcrawl.tasks.evaluation.molecule_nat_lang` | `workflows/eval-molecule-nat-lang.sh` |
| text_alignment | chebi20 | `molcrawl.tasks.evaluation.chebi20` | `workflows/eval-chebi20.sh` |
| text_alignment | chemllmbench | `molcrawl.tasks.evaluation.chemllmbench` | `workflows/eval-chemllmbench.sh` |

## 4. Smoke run

The Phase 0 smoke workflow verifies the whole `BaseEvaluator -> adapter
-> report` path against a small ClinVar sample.

```bash
export MODEL_PATH=runs_train_gpt2_genome_small/ckpt.pt
export TOKENIZER_PATH=assets/tokenizers/genome.model
export CLINVAR_DATA=learning_source/eval/clinvar/clinvar_small.csv
export MAX_EXAMPLES=16

bash workflows/eval-_smoke.sh
```

The run is successful when `experiment_data/eval/clinvar_smoke/metrics.json`
and `REPORT.md` have been written.

## 5. Per-task usage

### 5.1 ClinVar (variant_effect, genome)

```bash
python -m molcrawl.tasks.evaluation.clinvar \
  --model-path runs_train_gpt2_genome_small/ckpt.pt \
  --tokenizer-path assets/tokenizers/genome.model \
  --clinvar-data learning_source/eval/clinvar/clinvar.csv \
  --arch gpt2 --modality genome_sequence \
  --output-dir experiment_data/eval/clinvar
```

### 5.2 MoleculeNet (property_prediction, compounds)

The evaluator expects `LEARNING_SOURCE_DIR/eval/moleculenet/<subtask>/raw.csv`
together with a `manifest.json`.

```bash
export MODEL_PATH=runs_train_chemberta2_small
export MOLECULENET_DIR=learning_source/eval/moleculenet
export SUBTASKS="bbbp esol"
bash workflows/eval-moleculenet.sh
```

Direct invocation:

```bash
python -m molcrawl.tasks.evaluation.moleculenet \
  --model-path "$MODEL_PATH" \
  --arch chemberta2 --modality compounds \
  --subtask bbbp \
  --task-dir "$MOLECULENET_DIR/bbbp" \
  --output-dir experiment_data/eval/moleculenet/bbbp
```

### 5.3 MOSES (generation_quality, compounds)

```bash
python -m molcrawl.tasks.evaluation.moses \
  --model-path runs_train_gpt2_compounds_small/ckpt.pt \
  --tokenizer-path assets/molecules/spm.model \
  --arch gpt2 --modality compounds \
  --reference-dir learning_source/eval/moses \
  --num-samples 30000 \
  --output-dir experiment_data/eval/moses
```

When the upstream `moses` Python package is installed, extended metrics
(FCD, SNN, Fragment, Scaffold, ...) are added under the `moses.*`
namespace.  Environments without that dependency receive only the core
metrics (validity, uniqueness, novelty, internal_diversity).

### 5.4 TAPE (property_prediction / sequence_annotation, protein)

```bash
python -m molcrawl.tasks.evaluation.tape \
  --model-path runs_train_esm2_small/ckpt.pt \
  --arch esm2 --modality protein_sequence \
  --task fluorescence \
  --task-dir learning_source/eval/tape/fluorescence \
  --output-dir experiment_data/eval/tape/fluorescence
```

`contact_prediction` is a placeholder (returns NaN) until the PDB-level
labelling is wired in.

### 5.5 GUE 28 tasks

```bash
export MODEL_PATH=runs_train_dnabert2_small
export GUE_DIR=learning_source/eval/gue
bash workflows/eval-gue.sh
```

Partial runs are supported via the `TASKS` environment variable:

```bash
TASKS="prom_300_all H3K4me3 tf_0" bash workflows/eval-gue.sh
```

## 6. Config YAMLs

Each task carries sample configs under `configs/<arch>_<size>.yaml`.
The keys mirror CLI arguments; load them via pydantic / PyYAML and pass
to the evaluator as `config=`.

```yaml
# molcrawl/tasks/evaluation/moleculenet/configs/moleculenet_bbbp_chemberta2.yaml
task: moleculenet
subtask: bbbp
modality: compounds
arch: chemberta2
split: scaffold
val_frac: 0.1
test_frac: 0.1
seed: 0
```

## 7. Weekly snapshot (Phase 6)

Once every task writes its metrics under a common root, build the cross
-task snapshot with:

```bash
bash workflows/eval-report-weekly.sh
# Walks INPUT_DIR (default: experiment_data/eval) and
# writes snapshot_<YYYYMMDD>.json + snapshot_<YYYYMMDD>.md into
# OUTPUT_DIR (default: docs/evaluation).
```

To include deltas against a previous snapshot:

```bash
PREVIOUS=docs/evaluation/snapshot_20260415.json \
  bash workflows/eval-report-weekly.sh
```

The script performs three steps:

1. Collect every `metrics.json` below `INPUT_DIR`.
2. Deduplicate by `(modality, arch, task)` keeping the most recent run.
3. Render the 3-axis matrix and the top-20 movers vs. the previous
   snapshot in markdown.

## 8. Adding a new task

1. Create `molcrawl/tasks/evaluation/<task_name>/` with the six
   standard files (`data_preparation.py`, `splits.py`, `metrics.py`,
   `evaluator.py`, `visualization.py`, `__init__.py`, `__main__.py`).
2. Subclass `BaseEvaluator` and implement `task_name`, `category()`,
   `load_dataset()`, `run_predictions()`, `compute_metrics()`.
3. Drop a starter config under `configs/<arch>_<size>.yaml`.
4. Add a matching `workflows/eval-<task>.sh` that maps environment
   variables to CLI arguments.
5. Add a unit test to `tests/unit/test_tasks_evaluation_<phase>.py`.
   The pipeline can be exercised with a synthetic adapter registered
   via `register_adapter`.

## 9. Adding a new adapter

1. Create `molcrawl/tasks/evaluation/_adapters/<arch>_adapter.py` that
   subclasses `ModelAdapter`.
2. Override the capability methods you support (`embed`,
   `score_likelihood`, `generate`, etc.).
3. Call `register_adapter("<arch>", MyAdapter)` at the bottom of the
   module.
4. Add the import to `molcrawl/tasks/evaluation/_adapters/__init__.py`.

## 10. Known limitations

- Only the `gpt2` adapter is registered today.  Other architectures
  (bert, chemberta2, esm2, dnabert2) need adapters before
  their tasks can run on real data.
- `contact_prediction` (TAPE), `pfam_hit_rate`
  (protein_foldability), and the ChEBI-20 BLEU / ROUGE metrics rely on
  heavy optional dependencies (PDB tooling, HMMER, `nltk`,
  `rouge-score`) and currently return NaN placeholders when those are
  unavailable.
- There is no generic config-YAML driven runner yet - each task is
  invoked via its `__main__.py`.  A project-wide runner that ingests
  the YAMLs directly can be layered on top when needed.
