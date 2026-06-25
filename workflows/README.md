# Workflow Scripts

Workflow scripts for data preparation, model training, evaluation, and maintenance for the RIKEN Dataset Foundational Model project.

**Last Updated**: March 28, 2026
**Total Scripts**: 91 (Shell: 89, Python: 2)

## Table of Contents

- [Overview](#-overview)
- [Initial Setup](#-initial-setup)
- [Data Preparation Scripts](#-data-preparation-scripts)
- [Model Training Scripts](#-model-training-scripts)
- [AI Model Evaluation Scripts](#-ai-model-evaluation-scripts)
- [Development & Testing](#-development--testing)
- [Web Interface & Services](#-web-interface--services)
- [Output Structure](#-output-structure)
- [Quick Start Examples](#-quick-start-examples)
- [Prerequisites](#-prerequisites)
- [Script Categories](#-script-categories)
- [Integrated Script Structure](#-integrated-script-structure)
- [Important Notes](#-important-notes)
- [Troubleshooting](#-troubleshooting)
- [Migration Notes](#-migration-notes)

## Overview

This directory contains shell scripts for various project operations including data preparation, model training, evaluation, testing, and system maintenance. All scripts should be executed from the project root directory unless otherwise specified.

The workflow scripts are organized into several categories:

- **Data Preparation** (Phase 01-02): Dataset tokenization and format conversion - 17 scripts
- **Model Training** (Phase 03a-03g): GPT-2, BERT, DNABERT-2, ESM-2, ChemBERTa-2 - 43 scripts
- **Model Evaluation**: Comprehensive evaluation with visualization - 9 scripts
- **Development & Testing**: Debugging, batch testing, and validation tools - 6 scripts
- **System Infrastructure**: Web services, experiment tracking, and utilities - 4 scripts
- **Common Library**: Shared utility functions - 1 script

```bash
# Usage pattern
cd /path/to/riken-dataset-fundational-model
./workflows/script_name.sh
```

## 🛠️ Initial Setup

### Environment Setup

| Script        | Purpose                      | Function                                                           |
| ------------- | ---------------------------- | ------------------------------------------------------------------ |
| `00-first.sh` | First-time environment setup | Configure conda channels, create environment, install dependencies |

## 📊 Data Preparation Scripts

This section contains **17 data preparation scripts** (Phase 1: 11 scripts, Phase 2: 6 scripts)

### Phase 1: Dataset Preparation

| Script                                             | Purpose                                    | Model Type        | Output                          |
| -------------------------------------------------- | ------------------------------------------ | ----------------- | ------------------------------- |
| `01-compounds_prepare.sh`                          | Compounds (OrganiX13) dataset tokenization | compounds         | Tokenized SMILES/Scaffolds data |
| `01-compounds_chembl-prepare.sh`                   | ChEMBL 36 dataset preparation              | compounds         | ChEMBL tokenized data           |
| `01-compounds_guacamol-prepare.sh`                 | GuacaMol compounds preparation             | compounds         | GuacaMol benchmark data         |
| `01-genome_sequence-prepare.sh`                    | Genome sequence (RefSeq) data prep         | genome_sequence   | Tokenized genome sequences      |
| `01-genome_sequence_clinvar-prepare.sh`            | ClinVar variant dataset preparation        | genome_sequence   | ClinVar tokenized data          |
| `01-molecule_nat_lang-prepare.sh`                  | Molecule natural language (SMolInstruct)   | molecule_nat_lang | Molecule descriptions           |
| `01-molecule_nat_lang_mol_instructions-prepare.sh` | Mol-Instructions dataset preparation       | molecule_nat_lang | Mol-Instructions tokenized data |
| `01-protein_sequence-prepare.sh`                   | Protein sequence (UniRef50) data prep      | protein_sequence  | Tokenized protein sequences     |
| `01-protein_sequence_proteingym-prepare.sh`        | ProteinGym v1.3 DMS dataset preparation    | protein_sequence  | ProteinGym tokenized data       |
| `01-rna-prepare.sh`                                | RNA sequence (CELLxGENE) data preparation  | rna               | Tokenized RNA sequences         |
| `01-rna_celltype-prepare.sh`                       | Cell type annotation dataset preparation   | rna               | Geneformer cell type data       |

### Phase 2: GPT-2 Data Preparation

| Script                                       | Purpose                            | Model Type        | Function                |
| -------------------------------------------- | ---------------------------------- | ----------------- | ----------------------- |
| `02-compounds-prepare-gpt2.sh`               | GPT-2 compounds (OrganiX13) data   | compounds         | Convert to GPT-2 format |
| `02-compounds_organix13-prepare-gpt2.sh`     | GPT-2 OrganiX13 data (alternative) | compounds         | Convert to GPT-2 format |
| `02-genome_sequence-prepare-gpt2.sh`         | GPT-2 genome data                  | genome_sequence   | Convert to GPT-2 format |
| `02-molecule_nat_lang-prepare-gpt2.sh`       | GPT-2 molecule NL data             | molecule_nat_lang | Convert to GPT-2 format |
| `02-protein_sequence-prepare-gpt2.sh`        | GPT-2 protein data                 | protein_sequence  | Convert to GPT-2 format |
| `02-rna-prepare-gpt2.sh`                     | GPT-2 RNA data                     | rna               | Convert to GPT-2 format |

### Utility Scripts

| Script                                  | Purpose                      | Function                                                                              |
| --------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------- |
| `common_functions.sh`                   | Common function library      | Helper functions for GPU selection, memory check, and environment variable validation |
| `convert_molecule_nat_lang_to_arrow.sh` | Convert molecule data        | Convert to Arrow format                                                               |
| `create_sample_vocab.sh`                | Generate sample vocabulary   | Development setup                                                                     |

## 🏋️ Model Training Scripts

This section contains **46 training scripts** (Phase 3a: 29, Phase 3b: 1, Phase 3c: 11, Phase 3d: 3, Phase 3e: 3, Phase 3f: 3, Phase 3g: 3)

### Phase 3a: Standard GPT-2 Training

| Script                                                  | Purpose                     | Model Size | Training Type |
| ------------------------------------------------------- | --------------------------- | ---------- | ------------- |
| `03a-compounds-train-gpt2-small.sh`                     | Compounds (OrganiX13) GPT-2 | Small      | Standard      |
| `03a-compounds-train-gpt2-medium.sh`                    | Compounds (OrganiX13) GPT-2 | Medium     | Standard      |
| `03a-compounds-train-gpt2-large.sh`                     | Compounds (OrganiX13) GPT-2 | Large      | Standard      |
| `03a-compounds-train-gpt2-xl.sh`                        | Compounds (OrganiX13) GPT-2 | XL         | Standard      |
| `03a-compounds_chembl-train-gpt2-small.sh`              | ChEMBL compounds GPT-2      | Small      | Standard      |
| `03a-compounds_guacamol-train-small.sh`                 | GuacaMol compounds          | Small      | Standard      |
| `03a-compounds_guacamol-train-medium.sh`                | GuacaMol compounds          | Medium     | Standard      |
| `03a-compounds_guacamol-train-large.sh`                 | GuacaMol compounds          | Large      | Standard      |
| `03a-compounds_guacamol-train-xl.sh`                    | GuacaMol compounds          | XL         | Standard      |
| `03a-genome_sequence-train-small.sh`                    | Genome sequence (RefSeq)    | Small      | Standard      |
| `03a-genome_sequence-train-medium.sh`                   | Genome sequence (RefSeq)    | Medium     | Standard      |
| `03a-genome_sequence-train-large.sh`                    | Genome sequence (RefSeq)    | Large      | Standard      |
| `03a-genome_sequence-train-xl.sh`                       | Genome sequence (RefSeq)    | XL         | Standard      |
| `03a-genome_sequence_clinvar-train-gpt2-small.sh`       | ClinVar genome GPT-2        | Small      | Standard      |
| `03a-molecule_nat_lang-train-small.sh`                  | Molecule NL (SMolInstruct)  | Small      | Standard      |
| `03a-molecule_nat_lang-train-medium.sh`                 | Molecule NL (SMolInstruct)  | Medium     | Standard      |
| `03a-molecule_nat_lang-train-large.sh`                  | Molecule NL (SMolInstruct)  | Large      | Standard      |
| `03a-molecule_nat_lang-train-xl.sh`                     | Molecule NL (SMolInstruct)  | XL         | Standard      |
| `03a-molecule_nat_lang_mol_instructions-train-small.sh` | Mol-Instructions GPT-2      | Small      | Standard      |
| `03a-protein_sequence-train-small.sh`                   | Protein sequence (UniRef50) | Small      | Standard      |
| `03a-protein_sequence-train-medium.sh`                  | Protein sequence (UniRef50) | Medium     | Standard      |
| `03a-protein_sequence-train-large.sh`                   | Protein sequence (UniRef50) | Large      | Standard      |
| `03a-protein_sequence-train-xl.sh`                      | Protein sequence (UniRef50) | XL         | Standard      |
| `03a-protein_sequence_proteingym-train-gpt2-small.sh`   | ProteinGym GPT-2            | Small      | Standard      |
| `03a-rna-train-small.sh`                                | RNA sequence (CELLxGENE)    | Small      | Standard      |
| `03a-rna-train-medium.sh`                               | RNA sequence (CELLxGENE)    | Medium     | Standard      |
| `03a-rna-train-large.sh`                                | RNA sequence (CELLxGENE)    | Large      | Standard      |
| `03a-rna-train-xl.sh`                                   | RNA sequence (CELLxGENE)    | XL         | Standard      |
| `03a-rna_celltype-train-gpt2-small.sh`                  | Cell type annotation GPT-2  | Small      | Standard      |

### Phase 3b: Enhanced Training

| Script                                     | Purpose                         | Enhancement                  |
| ------------------------------------------ | ------------------------------- | ---------------------------- |
| `03b-genome_sequence-train-wandb-small.sh` | Genome training with monitoring | Weights & Biases integration |

### Phase 3c: BERT Model Training

| Script                                                       | Purpose                          | Model Size |
| ------------------------------------------------------------ | -------------------------------- | ---------- |
| `03c-compounds-train-bert-small.sh`                          | Compounds (OrganiX13) BERT       | Small      |
| `03c-compounds_chembl-train-bert-small.sh`                   | ChEMBL compounds BERT            | Small      |
| `03c-compounds_guacamol-train-bert-small.sh`                 | GuacaMol compounds BERT          | Small      |
| `03c-genome_sequence-train-bert-small.sh`                    | Genome sequence (RefSeq) BERT    | Small      |
| `03c-genome_sequence_clinvar-train-bert-small.sh`            | ClinVar genome BERT              | Small      |
| `03c-molecule_nat_lang-train-bert-small.sh`                  | Molecule NL (SMolInstruct) BERT  | Small      |
| `03c-molecule_nat_lang_mol_instructions-train-bert-small.sh` | Mol-Instructions BERT            | Small      |
| `03c-protein_sequence-train-bert-small.sh`                   | Protein sequence (UniRef50) BERT | Small      |
| `03c-protein_sequence_proteingym-train-bert-small.sh`        | ProteinGym BERT                  | Small      |
| `03c-rna-train-bert-small.sh`                                | RNA sequence (CELLxGENE) BERT    | Small      |
| `03c-rna_celltype-train-bert-small.sh`                       | Cell type annotation BERT        | Small      |

### Phase 3d: DNABERT-2 Training

| Script                                         | Purpose                   | Model Size |
| ---------------------------------------------- | ------------------------- | ---------- |
| `03d-genome_sequence-train-dnabert2-small.sh`  | Genome sequence DNABERT-2 | Small      |
| `03d-genome_sequence-train-dnabert2-medium.sh` | Genome sequence DNABERT-2 | Medium     |
| `03d-genome_sequence-train-dnabert2-large.sh`  | Genome sequence DNABERT-2 | Large      |

### Phase 3e: ESM-2 Training

| Script                                      | Purpose                | Model Size |
| ------------------------------------------- | ---------------------- | ---------- |
| `03e-protein_sequence-train-esm2-small.sh`  | Protein sequence ESM-2 | Small      |
| `03e-protein_sequence-train-esm2-medium.sh` | Protein sequence ESM-2 | Medium     |
| `03e-protein_sequence-train-esm2-large.sh`  | Protein sequence ESM-2 | Large      |

### Phase 3g: ChemBERTa-2 Training

| Script                                     | Purpose              | Model Size |
| ------------------------------------------ | -------------------- | ---------- |
| `03g-compounds-train-chemberta2-small.sh`  | Compounds ChemBERTa-2 | Small     |
| `03g-compounds-train-chemberta2-medium.sh` | Compounds ChemBERTa-2 | Medium    |
| `03g-compounds-train-chemberta2-large.sh`  | Compounds ChemBERTa-2 | Large     |

## 🚀 AI Model Evaluation Scripts

The evaluation harness has migrated to a single arch-agnostic layout
under `molcrawl/tasks/evaluation/<task>/`. Each task ships a
`__main__.py` CLI plus a thin `workflows/eval-<task>.sh` driver, and
data fetches live under `workflows/data/eval-data-<task>.sh`.
Credential-gated downloads (COSMIC, OMIM, gated HuggingFace datasets)
read from a repo-root `.env` — see [`.env.example`](../.env.example)
for the full list.

| Workflow                              | Task package                                            | Modality          | Notes                                       |
| ------------------------------------- | ------------------------------------------------------- | ----------------- | ------------------------------------------- |
| `eval-clinvar.sh`                     | `molcrawl.tasks.evaluation.clinvar`                     | genome_sequence   | balanced positive/negative + bootstrap CI   |
| `eval-gnomad.sh`                      | `molcrawl.tasks.evaluation.gnomad_af_correlation`       | genome_sequence   | AF-bin stratified                           |
| `eval-gue.sh`                         | `molcrawl.tasks.evaluation.gue`                         | genome_sequence   | 28 sub-tasks                                |
| `eval-proteingym.sh`                  | `molcrawl.tasks.evaluation.proteingym`                  | protein_sequence  | reference-vs-variant likelihood             |
| `eval-deeploc.sh`                     | `molcrawl.tasks.evaluation.deeploc`                     | protein_sequence  | 10-class subcellular localisation           |
| `eval-tape.sh`                        | `molcrawl.tasks.evaluation.tape`                        | protein_sequence  | fluorescence / stability / remote_homology / SS3 / SS8 |
| `eval-protein-foldability.sh`         | `molcrawl.tasks.evaluation.protein_foldability`         | protein_sequence  | structure-free foldability proxies          |
| `eval-moleculenet.sh`                 | `molcrawl.tasks.evaluation.moleculenet`                 | compounds         | 12 standard property-prediction subsets     |
| `eval-moses.sh`                       | `molcrawl.tasks.evaluation.moses`                       | compounds         | validity / uniqueness / novelty / int. div. |
| `eval-chembl-heldout.sh`              | `molcrawl.tasks.evaluation.chembl_scaffold_heldout`     | compounds         | scaffold-disjoint perplexity                |
| `eval-rna-benchmark.sh`               | `molcrawl.tasks.evaluation.rna_benchmark`               | rna               | per-tissue PLL                              |
| `eval-tabula-sapiens.sh`              | `molcrawl.tasks.evaluation.tabula_sapiens`              | rna               | cell-type annotation                        |
| `eval-replogle-perturb-seq.sh`        | `molcrawl.tasks.evaluation.replogle_perturb_seq`        | rna               | perturbation response                       |
| `eval-molecule-nat-lang.sh`           | `molcrawl.tasks.evaluation.molecule_nat_lang`           | molecule_nat_lang | molecule / caption pair likelihood          |
| `eval-chemllmbench.sh`                | `molcrawl.tasks.evaluation.chemllmbench`                | molecule_nat_lang | 9 sub-tasks (3 currently wired)             |
| `eval-chebi20.sh`                     | `molcrawl.tasks.evaluation.chebi20`                     | molecule_nat_lang | bidirectional generation                    |
| (credential-gated) `eval-data-cosmic.sh` | `molcrawl.tasks.evaluation.cosmic`                   | genome_sequence   | requires COSMIC_EMAIL / COSMIC_PASSWORD     |
| (credential-gated) `eval-data-omim.sh`   | `molcrawl.tasks.evaluation.omim`                     | genome_sequence   | requires OMIM_API_KEY                       |

**Common features**:

- **Bootstrap 95 % CIs** rendered alongside point estimates wherever the
  metric is well-defined under resampling.
- **Predictions log**: every evaluator emits `predictions.jsonl`
  (per-row records) and `predictions.txt` (best/worst-fit narrative)
  next to `metrics.json` and `REPORT.md`.
- **Idempotent matrix runner**: `workflows/eval-matrix-bench.sh` drives
  a full (evaluator × arch × size) sweep and skips combos whose
  REPORT.md already exists.
- **Dashboard**: `python -m molcrawl.tasks.evaluation._dashboard` walks
  the REPORT.md files and rebuilds `docs-src/assets/data/evaluations.json`
  for the static dashboard.

The pre-refactor per-architecture wrappers (`run_bert_*`, `run_gpt2_*`)
have been retired in favour of this layout. The legacy
`protein_classification` task is also gone — its use cases are covered
by `proteingym` (variant effect), `deeploc` (localisation), and
`tape.remote_homology` (fold classification).

## 🔧 Development & Testing

### Testing Scripts

| Script                    | Purpose                   | Function                                                                                                                           |
| ------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `batch_test_gpt2.sh`      | GPT-2 model batch testing | Automatically finds and batch-tests checkpoints across multiple domains (compounds, molecule_nat_lang, genome, protein_sequence, rna) |
| `gpt2_test_checkpoint.sh` | GPT-2 checkpoint validation | Model checkpoint testing                                                                                                         |
| `debug_protein_bert.sh`   | BERT protein model debugging | Troubleshooting training issues                                                                                                  |

### System Utilities

| Script                  | Purpose                | Function                  |
| ----------------------- | ---------------------- | ------------------------- |
| `reboot-cause-check.sh` | System reboot analysis | Infrastructure monitoring |

## 🏗️ Web Interface & Services

This section contains **5 scripts** (Web: 2, Experiment Management: 3)

### Web Interface

| Script                | Purpose                 | Function                          | Port/Service  |
| --------------------- | ----------------------- | --------------------------------- | ------------- |
| `web.sh`              | Launch web interface    | Dataset browser and visualization | Default: 3001 |
| `start_api_server.py` | Web API for experiments | RESTful service                   | Default: 8000 |

### Experiment Management

| Script                       | Purpose                        | Function              |
| ---------------------------- | ------------------------------ | --------------------- |
| `setup_experiment_system.sh` | Initialize experiment tracking | System configuration  |
| `start_experiment_system.sh` | Launch experiment services     | Service orchestration |
| `demo_experiment_system.sh`  | System demonstration           | Testing & validation  |

## 📊 Output Structure

All evaluation scripts use a structured directory format and support custom output directories.

### Environment Variables

Evaluation scripts use the following environment variables:

| Variable                | Purpose                          | Default                | Required |
| ----------------------- | -------------------------------- | ---------------------- | -------- |
| `LEARNING_SOURCE_DIR`   | Input data directory (read-only) | —                      | ✅       |
| `EVALUATION_OUTPUT_DIR` | Output data directory (writable) | `$LEARNING_SOURCE_DIR` | ❌       |

**Usage examples**:

```bash
# Initial setup
./workflows/00_first.sh

# Basic data preparation flow
export LEARNING_SOURCE_DIR=/data/learning_source

# Phase 1: Dataset preparation
./workflows/01_compounds_prepare.sh
./workflows/01_genome-sequence_prepare.sh
./workflows/01_protein-sequence_prepare.sh
# ... other datasets

# Phase 2: GPT-2 format conversion (if needed)
./workflows/02-compounds-prepare-gpt2.sh
# ... corresponding GPT-2 preparation scripts

# Phase 3: Training (optional)
./workflows/03a-compounds-guacamol-train-small.sh
# ... corresponding training scripts

# Evaluation (arch-agnostic harness)
bash workflows/data/eval-data-clinvar.sh   # download data
bash workflows/eval-clinvar.sh             # run evaluator

# Web interface
./workflows/web.sh

# Separating input and output
export LEARNING_SOURCE_DIR=/readonly/learning_source  # input (read-only)
export OUTPUT_DIR=/writable/outputs/clinvar           # eval output (writable)
bash workflows/eval-clinvar.sh
```

### Directory Structure

```
$LEARNING_SOURCE_DIR/                   # Learning data directory
├── compounds/                          # Compound data
│   ├── image/                          # Visualization images
│   └── data/                           # Tokenized data
├── genome_sequence/                    # Genome sequence data
│   ├── image/                          # Visualization images
│   ├── data/                           # Tokenized data
│   │   ├── clinvar/                    # ClinVar data
│   │   ├── cosmic/                     # COSMIC data
│   │   └── omim/                       # OMIM data
│   └── report/                         # Evaluation results
│       ├── bert_clinvar_evaluation/
│       ├── clinvar_evaluation/
│       └── cosmic_evaluation/
├── protein_sequence/                   # Protein sequence data
│   ├── image/                          # Visualization images
│   ├── data/                           # Tokenized data
│   └── report/                         # Evaluation results
│       ├── bert_proteingym/
│       └── gpt2_proteingym/
├── rna/                                # RNA sequence data
│   ├── image/                          # Visualization images
│   └── data/                           # Tokenized data
└── molecule_nat_lang/                  # Molecule natural language data
    ├── image/                          # Visualization images
    └── data/                           # Tokenized data

# Trained model outputs
gpt2-output/                            # GPT-2 model outputs
├── compounds-small/
├── genome_sequence-small/
├── protein_sequence-small/
└── rna-small/

# Execution logs
logs/                                   # Script execution logs
└── *.log                               # Log files per script
```

### Output Directory Customization

Every evaluation wrapper now composes its `OUTPUT_DIR` automatically as

```
${LEARNING_SOURCE_DIR}/experiment_data/eval/<modality>-<arch>-<size>/<RUNTAG>/
```

where the model slug is derived from `MODEL_PATH`. Operators only need
to set `RUNTAG` to keep historical results separated:

```bash
# ClinVar — gives you
#   ${LEARNING_SOURCE_DIR}/experiment_data/eval/genome_sequence-bert-small/clinvar_nper1000/
LEARNING_SOURCE_DIR=$LSD \
MODEL_PATH=$LSD/genome_sequence/bert-output/genome_sequence-small/checkpoint-60000 \
CLINVAR_DATA=$LSD/eval/clinvar/clinvar.csv \
RUNTAG=clinvar_nper1000 N_PER_CLASS=1000 \
bash workflows/eval-clinvar.sh

# ProteinGym, MOSES, MoleculeNet, DeepLoc, TAPE, GUE, ... all follow
# the same pattern. Override OUTPUT_DIR explicitly to escape the slug
# layout (rarely needed).
```

**Notes**:

- The model-slug parent is computed by `derive_model_slug` in
  `common_functions.sh` from `MODEL_PATH`; absolute and repo-relative
  paths both work.
- `RUNTAG` defaults to `<task>_default` to flag unspecified runs —
  set it explicitly per run to avoid clobbering historical output.
- All evaluators emit `metrics.json`, `REPORT.md`, `predictions.jsonl`,
  and `predictions.txt` under `OUTPUT_DIR`.
- Smoke runs land under `_smoke/<model-slug>/<RUNTAG>/`, failed runs
  (no `metrics.json`) under `_failed/<old_dir>/`.
- See [`docs/04-evaluation/eval_dashboard.ja.md`](../docs/04-evaluation/eval_dashboard.ja.md)
  for the full convention plus migration helpers.

### Evaluation Directory Contents

- `*_results.json` — Structured evaluation results
- `*_report.txt` — Human-readable summary
- `*_detailed_results.csv` — Per-sample predictions
- `visualizations/` — Charts and graphs generated during the visualization phase

## 🎯 Quick Start Examples

### Standard Evaluations

The arch-agnostic evaluators all share the same `eval-<task>.sh`
wrapper convention. Each takes an env-var-style configuration:

```bash
# ClinVar — pathogenic/benign variant classification (genome decoder)
MODEL_PATH=$LEARNING_SOURCE_DIR/genome_sequence/gpt2-output/.../ckpt.pt \
TOKENIZER_PATH=assets/genome/sentencepiece.model \
CLINVAR_CSV=$LEARNING_SOURCE_DIR/eval/clinvar/clinvar_sequences.csv \
N_PER_CLASS=1000 BOOTSTRAP=100 \
bash workflows/eval-clinvar.sh

# ProteinGym — zero-shot variant effect (protein encoder)
MODEL_PATH=$LEARNING_SOURCE_DIR/protein_sequence/esm2-output/.../checkpoint-25000 \
PROTEINGYM_DIR=$LEARNING_SOURCE_DIR/eval/proteingym/unpacked \
ARCH=esm2 MAX_EXAMPLES=300 BOOTSTRAP=100 \
bash workflows/eval-proteingym.sh

# DeepLoc — subcellular localisation (10-class probe over esm2 / bert)
MODEL_PATH=$LEARNING_SOURCE_DIR/protein_sequence/esm2-output/.../checkpoint-25000 \
DEEPLOC_DATA=$LEARNING_SOURCE_DIR/eval/deeploc/deeploc.csv \
ARCH=esm2 MAX_EXAMPLES=300 BOOTSTRAP=100 \
bash workflows/eval-deeploc.sh

# TAPE — fluorescence / stability / remote_homology / SS3 / SS8
MODEL_PATH=$LEARNING_SOURCE_DIR/protein_sequence/esm2-output/.../checkpoint-25000 \
TAPE_DIR=$LEARNING_SOURCE_DIR/eval/tape \
TASKS="fluorescence stability remote_homology" \
ARCH=esm2 MAX_EXAMPLES=300 BOOTSTRAP=100 \
bash workflows/eval-tape.sh
```

The full task list is in the table earlier in this README.

### Matrix sweep (one command across all evaluators)

```bash
MATRIX_DEVICE=cpu MATRIX_BOOTSTRAP=30 \
bash workflows/eval-matrix-bench.sh
```

`eval-matrix-bench.sh` is idempotent — combos whose `REPORT.md`
already exists are skipped on re-runs. Use `MATRIX_FILTER=<regex>` to
restrict to a sub-set of `<task>__<arch>__<size>` ids.

### Custom output directory

Every wrapper honours `OUTPUT_DIR=...`. Default is
`${LEARNING_SOURCE_DIR}/experiment_data/eval/<modality>-<arch>-<size>/<RUNTAG>/`
(model slug derived from `MODEL_PATH`). `MAX_EXAMPLES`, `BOOTSTRAP`,
`PREDICTIONS_PREVIEW_COUNT`, `SEED`, and `RUNTAG` are accepted by every
wrapper.

### Experiment System

```bash
# Complete system setup
./workflows/setup_experiment_system.sh

# Start all services
./workflows/start_experiment_system.sh

# Demo the system
./workflows/demo_experiment_system.sh
```

### Development Workflow

```bash
# Batch test all GPT-2 checkpoints
./workflows/batch_test_gpt2.sh gpt2-output/

# Test specific GPT-2 checkpoint
./workflows/gpt2_test_checkpoint.sh

# Debug BERT training
./workflows/debug_protein_bert.sh

# Create development vocabularies
./workflows/create_sample_vocab.sh
```

## 🔧 Prerequisites

### Common Functions Library

`common_functions.sh` provides shared utility functions used across multiple bootstrap scripts:

**Main Features**:

- `check_learning_source_dir()` — Validates the `LEARNING_SOURCE_DIR` environment variable
- `select_best_gpu()` — Automatically selects the GPU with the most available memory
- `check_gpu_memory(gpu_id, min_memory_gb)` — Checks available GPU memory
- Other error handling and logging utilities

**Usage**:

```bash
# Load from another script
source "$(dirname "$0")/common_functions.sh"

# Check environment variable
check_learning_source_dir

# Select the best GPU
BEST_GPU=$(select_best_gpu)
export CUDA_VISIBLE_DEVICES=$BEST_GPU
```

### Environment Setup

```bash
# Required environment variables
export LEARNING_SOURCE_DIR=/path/to/learning_source_202508
export CUDA_VISIBLE_DEVICES=0  # For GPU usage

# Load project configuration
source molcrawl/core/env.sh
```

### Dependencies

- Python 3.8+ with transformers, torch, pandas, numpy
- CUDA-capable GPU for model training/evaluation
- Sufficient disk space for datasets and results
- Access to model checkpoints in appropriate directories

## 📝 Script Categories

This directory contains 91 scripts (Shell: 89, Python: 2):

### 🔍 **Evaluation Scripts** (9 scripts)

Arch-agnostic evaluation entry points under `workflows/eval-*.sh`
backed by Python packages under `molcrawl/tasks/evaluation/<task>/`.

The full list (data downloader → evaluator wrapper → task package) is
in the table earlier in this README. Highlights:

- `eval-clinvar.sh` — ClinVar pathogenic / benign variant classification
- `eval-gnomad.sh` — gnomAD allele-frequency correlation
- `eval-gue.sh` — GUE 28-task genome benchmark
- `eval-proteingym.sh` — ProteinGym zero-shot variant effect
- `eval-deeploc.sh` — DeepLoc subcellular localisation
- `eval-tape.sh` — TAPE (fluorescence / stability / remote_homology / SS3 / SS8)
- `eval-protein-foldability.sh` — structure-free foldability proxies
- `eval-moleculenet.sh`, `eval-moses.sh`, `eval-chembl-heldout.sh`
- `eval-rna-benchmark.sh`, `eval-tabula-sapiens.sh`, `eval-replogle-perturb-seq.sh`
- `eval-molecule-nat-lang.sh`, `eval-chemllmbench.sh`, `eval-chebi20.sh`
- (credential-gated) `eval-data-cosmic.sh`, `eval-data-omim.sh` — paired
  with `molcrawl.tasks.evaluation.{cosmic,omim}` and `.env`

### 🛠️ **Development Scripts** (4 scripts)

Debugging, testing, and development utilities

- `batch_test_gpt2.sh` - GPT-2 checkpoint batch testing (all domains)
- `gpt2_test_checkpoint.sh` - GPT-2 checkpoint validation
- `debug_protein_bert.sh` - BERT model debugging
- `reboot-cause-check.sh` - System reboot cause analysis

### 🏭 **Infrastructure Scripts** (4 scripts)

System setup, service management, and experiment tracking infrastructure

- `setup_experiment_system.sh` - Initialize experiment system
- `start_experiment_system.sh` - Start experiment services
- `demo_experiment_system.sh` - System demonstration
- `start_api_server.py` - Start web API server

### ⚙️ **Utility Scripts** (2 scripts)

Helper scripts for data preparation and project setup

- `common_functions.sh` - Common function library (GPU selection, memory check, environment variable validation)
- `create_sample_vocab.sh` - Generate sample vocabulary files

## 🔄 Integrated Script Structure

### Three-Phase Pipeline

All evaluation scripts are structured around three phases:

1. **Data Preparation Phase** (skippable with `--skip_data_prep`)
   - Dataset download/generation
   - Preprocessing and format conversion
   - Saved to `$LEARNING_SOURCE_DIR/{model_type}/data/`
   - **Customization**: Some scripts support `--data_dir` option

2. **Model Evaluation Phase** (skippable with `--skip_evaluation`)
   - Load trained model
   - Run inference on dataset
   - Calculate metrics and save results
   - **Customization**: All scripts support `-o` or `--output_dir`

3. **Visualization Phase** (skippable with `--skip_visualization`)
   - Generate charts from evaluation results
   - Create HTML reports
   - Saved to `{output_dir}/visualizations/` subdirectory
   - **Customization**: Visualization scripts support `--output_dir`

### Flexible Output Directory

Every evaluation wrapper takes `OUTPUT_DIR=...` as its output target.
Default is `${LEARNING_SOURCE_DIR}/experiment_data/eval/<modality>-<arch>-<size>/<RUNTAG>/`
(model-first layout — see Output Directory Customization above).

```bash
# Default (slug derived from MODEL_PATH; RUNTAG defaults to clinvar_default)
bash workflows/eval-clinvar.sh
# -> ${LEARNING_SOURCE_DIR}/experiment_data/eval/genome_sequence-<arch>-<size>/clinvar_default/

# Set RUNTAG to keep historical runs separated
RUNTAG=clinvar_nper1000 bash workflows/eval-clinvar.sh
# -> ${LEARNING_SOURCE_DIR}/experiment_data/eval/<slug>/clinvar_nper1000/

# Override OUTPUT_DIR to escape the slug layout entirely
OUTPUT_DIR=/mnt/results/my_clinvar bash workflows/eval-clinvar.sh
# -> /mnt/results/my_clinvar/
```

Each run writes `metrics.json`, `REPORT.md`, `predictions.jsonl`, and
`predictions.txt` into `OUTPUT_DIR`.

### Benefits of Phase-Based Execution

- **Development efficiency**: Prepare data once, iterate over evaluation and visualization
- **Easy debugging**: Test each phase independently
- **Resource management**: Run only the phases needed to save resources
- **Flexibility**: Skip data preparation when using externally prepared data

## 🚨 Important Notes

### Execution Environment

- **Execution location**: All scripts should be run from the project root directory
- **LEARNING_SOURCE_DIR**: Required environment variable used by all evaluation scripts
- **GPU requirements**: CUDA-capable GPU recommended (CPU execution is possible but slower)

### Data Management

- **Output management**: Results are automatically organized with timestamps
- **Credential-gated downloads**: COSMIC and OMIM data fetches need API
  credentials — set them in the repo-root `.env` (see `.env.example`)
  and run `workflows/data/eval-data-{cosmic,omim}.sh`.

### Script Structure

- **Integrated scripts**: Three phases (data preparation, evaluation, visualization) combined into a single script
- **Phase skipping**: Use `--skip_*` options to skip any phase
- **Error handling**: Robust error checking and recovery features

### Resource Management

- **GPU memory**: Varies with model size and batch size
- **Disk space**: Account for dataset and result file sizes
- **Logging**: Comprehensive logging for all operations
- **Performance**: GPU usage is approx. 4× faster than CPU (e.g., ProteinGym 50 samples/GPU ≈ 12 seconds)

### Performance Optimization

- **Default device**: every evaluator wrapper accepts `DEVICE=cuda` or
  `DEVICE=cpu` (default `cuda`).
- **Sample limit**: `MAX_EXAMPLES=N` to cap how many rows are scored.
- **Bootstrap iterations**: `BOOTSTRAP=N` (default 100) to control CI
  estimation cost; set to `0` to disable.
- **Idempotent reruns**: the matrix runner skips combos whose REPORT.md
  already exists; individual wrappers don't, so set `OUTPUT_DIR` to a
  fresh path when re-running.

## 📞 Troubleshooting

### Common Issues and Solutions

1. **`LEARNING_SOURCE_DIR` not set**

   ```bash
   export LEARNING_SOURCE_DIR=/path/to/learning_source
   ```

2. **Model file not found** — every wrapper takes `MODEL_PATH=...`. The
   accepted layouts are `<dir>/ckpt.pt` for GPT-2 / minGPT checkpoints
   and `<dir>/checkpoint-N/` for HuggingFace MLM checkpoints.

3. **Data file not found** — re-run the corresponding
   `workflows/data/eval-data-<task>.sh` downloader. Each one is
   idempotent.

4. **Credential-gated downloads (COSMIC / OMIM / gated HuggingFace)**

   ```bash
   # Verify credentials are loaded
   grep -E '^(COSMIC_|OMIM_API_KEY|HF_TOKEN)' .env

   # Re-run the relevant downloader
   bash workflows/data/eval-data-cosmic.sh
   bash workflows/data/eval-data-omim.sh
   ```

5. **Slow evaluator on CPU** — reduce `MAX_EXAMPLES` (100–500 is typical
   for smoke runs) and `BOOTSTRAP` (20–30 keeps cost down).

6. **Missing Python packages**

   ```bash
   pip install torch transformers pandas numpy scikit-learn \
     matplotlib seaborn sentencepiece scipy huggingface_hub
   ```

### Output Directory Layout

```
$OUTPUT_DIR/
  metrics.json        # machine-readable metrics + bootstrap_ci_95
  REPORT.md           # human-readable summary
  predictions.jsonl   # per-row records
  predictions.txt     # narrative best/worst-fit preview
```

## 🔄 Migration Notes

The pre-refactor per-architecture wrappers (`run_bert_*`, `run_gpt2_*`)
and per-architecture evaluation modules (`bert_*.py`, `gpt2_*.py` under
each evaluator package) have been retired. The current arch-agnostic
pipeline:

1. Each evaluator lives under `molcrawl/tasks/evaluation/<task>/`.
2. Architectures (GPT-2 / BERT / ESM-2 / DNABERT-2 /
   ChemBERTa-2) plug in through `molcrawl/tasks/evaluation/_adapters/`.
3. Data fetches live in `workflows/data/eval-data-<task>.sh`.
4. Wrapper drivers live in `workflows/eval-<task>.sh`.
5. Credentials come from a single `.env` (template: `.env.example`).
