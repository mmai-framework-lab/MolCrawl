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
- **Model Training** (Phase 03a-03g): GPT-2, BERT, DNABERT-2, ESM-2, RNAformer, ChemBERTa-2 - 46 scripts
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

### Phase 3f: RNAformer Training

| Script                              | Purpose                | Model Size |
| ----------------------------------- | ---------------------- | ---------- |
| `03f-rna-train-rnaformer-small.sh`  | RNA sequence RNAformer | Small      |
| `03f-rna-train-rnaformer-medium.sh` | RNA sequence RNAformer | Medium     |
| `03f-rna-train-rnaformer-large.sh`  | RNA sequence RNAformer | Large      |

### Phase 3g: ChemBERTa-2 Training

| Script                                     | Purpose              | Model Size |
| ------------------------------------------ | -------------------- | ---------- |
| `03g-compounds-train-chemberta2-small.sh`  | Compounds ChemBERTa-2 | Small     |
| `03g-compounds-train-chemberta2-medium.sh` | Compounds ChemBERTa-2 | Medium    |
| `03g-compounds-train-chemberta2-large.sh`  | Compounds ChemBERTa-2 | Large     |

## 🚀 AI Model Evaluation Scripts

### BERT Model Evaluations

| Script                              | Purpose                                      | Dataset    | Dataset Size                            | Output Location                                                  |
| ----------------------------------- | -------------------------------------------- | ---------- | --------------------------------------- | ---------------------------------------------------------------- |
| `run_bert_proteingym_evaluation.sh` | BERT protein fitness prediction (integrated) | ProteinGym | Variable                                | `$LEARNING_SOURCE_DIR/protein_sequence/report/bert_proteingym_*` |
| `run_bert_clinvar_evaluation.sh`    | BERT variant pathogenicity prediction        | ClinVar    | 2,000 (1,000 positive + 1,000 negative) | `$LEARNING_SOURCE_DIR/genome_sequence/report/bert_clinvar_*`     |

**Notes**:

- The BERT ProteinGym script is a single integrated pipeline covering three phases: data preparation, evaluation, and visualization
- **ClinVar balanced sampling**: Randomly samples 1,000 pathogenic and 1,000 benign variants to ensure a balanced evaluation dataset

### GPT-2 Model Evaluations

#### Genome Sequence

| Script                              | Purpose                              | Dataset | Dataset Size                            | Default Device | Output Location                                                    |
| ----------------------------------- | ------------------------------------ | ------- | --------------------------------------- | -------------- | ------------------------------------------------------------------ |
| `run_gpt2_clinvar_evaluation.sh`    | Pathogenic variant prediction        | ClinVar | 2,000 (1,000 positive + 1,000 negative) | GPU (cuda)     | `$LEARNING_SOURCE_DIR/genome_sequence/report/clinvar_*`            |
| `run_gpt2_cosmic_evaluation.sh`     | Cancer-related variant analysis      | COSMIC  | Sample                                  | GPU (cuda)     | `$LEARNING_SOURCE_DIR/genome_sequence/report/cosmic_*`             |
| `run_gpt2_omim_evaluation_dummy.sh` | Hereditary disease prediction (dev)  | OMIM    | Sample                                  | GPU (cuda)     | `$LEARNING_SOURCE_DIR/genome_sequence/report/omim_evaluation`      |
| `run_gpt2_omim_evaluation_real.sh`  | Hereditary disease prediction (prod) | OMIM    | Real data                               | GPU (cuda)     | `$LEARNING_SOURCE_DIR/genome_sequence/report/omim_real_evaluation` |

**Notes**:

- `_dummy.sh`: Uses sample data for quick development and testing
- `_real.sh`: For production evaluation. Fetches real data from OMIM official database (authentication required)
- **GPU optimization**: All scripts use GPU (cuda) by default (approx. 4× faster than CPU)
- **Reuse existing data**: `run_gpt2_omim_evaluation_real.sh` supports `--existing_omim_dir` to skip re-downloading
- **ClinVar balanced sampling**: Randomly samples 1,000 pathogenic and 1,000 benign variants for balanced evaluation

#### Protein Sequence

| Script                               | Purpose                                      | Dataset    | Default Model          | Default Device | Output Location                                                            |
| ------------------------------------ | -------------------------------------------- | ---------- | ---------------------- | -------------- | -------------------------------------------------------------------------- |
| `run_gpt2_proteingym_evaluation.sh`  | Protein fitness prediction (integrated)      | ProteinGym | Required               | GPU (cuda)     | `$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_proteingym`             |
| `run_gpt2_protein_classification.sh` | Protein sequence classification (integrated) | Custom     | protein_sequence-small | GPU (cuda)     | `$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_protein_classification` |

**Notes**:

- **Integrated script**: A single script covering three phases: data preparation, evaluation, and visualization
- **Default model**: `run_gpt2_protein_classification.sh` can run without specifying a model (uses `gpt2-output/protein_sequence-small/ckpt.pt`)
- **Auto sample creation**: `run_gpt2_proteingym_evaluation.sh --create-sample` automatically downloads the recommended dataset
- **GPU optimization**: Uses GPU by default; switch to CPU with `--device cpu`
- **Rich visualization**: Automatically generates 10+ charts and detailed HTML reports

#### RNA Sequence

| Script                            | Purpose                  | Dataset       | Default Device | Output Location                                   |
| --------------------------------- | ------------------------ | ------------- | -------------- | ------------------------------------------------- |
| `run_rna_benchmark_evaluation.sh` | RNA benchmark evaluation | RNA Benchmark | GPU (cuda)     | `$LEARNING_SOURCE_DIR/rna/report/rna_benchmark_*` |

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

# Evaluation (standard usage)
./workflows/run_bert_clinvar_evaluation.sh --prepare-data

# Web interface
./workflows/web.sh

# Separating input and output
export LEARNING_SOURCE_DIR=/readonly/learning_source  # input (read-only)
export EVALUATION_OUTPUT_DIR=/writable/outputs        # output (writable)
./workflows/run_bert_clinvar_evaluation.sh --prepare-data
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

All evaluation scripts support custom output paths via `-o` or `--output_dir`:

```bash
# BERT ProteinGym evaluation - custom output
./workflows/run_bert_proteingym_evaluation.sh \
  --output_dir /custom/path/bert_proteingym_results

# GPT-2 ClinVar evaluation - custom output
./workflows/run_gpt2_clinvar_evaluation.sh \
  --output_dir /custom/path/clinvar_results

# GPT-2 ProteinGym evaluation - custom output
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m model.pt -d data.csv \
  -o /custom/path/proteingym_results

# GPT-2 OMIM real data evaluation - custom output
./workflows/run_gpt2_omim_evaluation_real.sh \
  --output_dir /custom/path/omim_real_results
```

**Notes**:

- If no output path is specified, results are saved to `$LEARNING_SOURCE_DIR/{model_type}/report/{evaluation_type}` by default
- The data preparation path (`--data_dir`) and report/visualization path (`--output_dir`) can be specified independently
- Visualization results are saved to the `{output_dir}/visualizations/` subdirectory

### Evaluation Directory Contents

- `*_results.json` — Structured evaluation results
- `*_report.txt` — Human-readable summary
- `*_detailed_results.csv` — Per-sample predictions
- `visualizations/` — Charts and graphs generated during the visualization phase

## 🎯 Quick Start Examples

### Standard Evaluations

#### BERT Model Evaluations

```bash
# BERT ProteinGym evaluation (integrated: data prep -> evaluation -> visualization)
./workflows/run_bert_proteingym_evaluation.sh --max_variants 2000 --batch_size 32

# Create sample data only
./workflows/run_bert_proteingym_evaluation.sh --sample_only

# Run evaluation only (skip data preparation)
./workflows/run_bert_proteingym_evaluation.sh --skip_data_prep

# BERT ClinVar evaluation (balanced sampling: 1,000 positive + 1,000 negative)
# First run: start from data preparation
./workflows/run_bert_clinvar_evaluation.sh --prepare-data

# If data is already prepared: run evaluation only
./workflows/run_bert_clinvar_evaluation.sh

# Force re-download data
./workflows/run_bert_clinvar_evaluation.sh --force-download
```

#### GPT-2 Genome Sequence Evaluations

```bash
# ClinVar evaluation (balanced sampling: 1,000 positive + 1,000 negative)
# First run: download data & balanced sampling
./workflows/run_gpt2_clinvar_evaluation.sh --download --model-size medium

# If data is already prepared: run evaluation only
./workflows/run_gpt2_clinvar_evaluation.sh --model-size small

# Run evaluation only (skip data preparation)
./workflows/run_gpt2_clinvar_evaluation.sh --eval-only --model-size medium

# Run visualization only
./workflows/run_gpt2_clinvar_evaluation.sh --visualize-only

# COSMIC evaluation
./workflows/run_gpt2_cosmic_evaluation.sh --model_size small --batch_size 32

# OMIM evaluation (sample data, for development)
./workflows/run_gpt2_omim_evaluation_dummy.sh --max_samples 50

# OMIM evaluation (real data, for production; authentication required)
./workflows/run_gpt2_omim_evaluation_real.sh --force_download --model_size medium

# OMIM evaluation (reuse existing data)
./workflows/run_gpt2_omim_evaluation_real.sh \
  --existing_omim_dir /path/to/downloaded/omim_data \
  --model_size medium
```

#### GPT-2 Protein Sequence Evaluations

```bash
# ProteinGym evaluation (integrated)
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m gpt2-output/protein_sequence-small/ckpt.pt \
  -d proteingym_data/sample.csv

# Auto-create sample data and evaluate (downloads recommended dataset)
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m gpt2-output/protein_sequence-small/ckpt.pt \
  --create-sample

# Protein Classification evaluation (using default model)
./workflows/run_gpt2_protein_classification.sh -s

# Protein Classification evaluation (custom model)
./workflows/run_gpt2_protein_classification.sh \
  -m gpt2-output/protein_sequence-medium/ckpt.pt \
  -s

# Run visualization only (if evaluation is already complete)
./workflows/run_gpt2_protein_classification.sh \
  -s --skip_data_prep --skip_evaluation
```

### Advanced Options

#### Phase-based Execution (GPT-2 scripts)

```bash
# Data preparation only
./workflows/run_gpt2_omim_evaluation_dummy.sh --skip_evaluation --skip_visualization

# Evaluation only (if data is already prepared)
./workflows/run_gpt2_omim_evaluation_dummy.sh --skip_data_prep --skip_visualization

# Visualization only (if evaluation results exist)
./workflows/run_gpt2_omim_evaluation_dummy.sh --skip_data_prep --skip_evaluation
```

#### Device and Performance Tuning

```bash
# Use CPU (for environments without GPU)
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m model.pt -d data.csv --device cpu

# Reduce batch size and sample count (to save memory)
./workflows/run_gpt2_clinvar_evaluation.sh \
  --max_samples 200 --batch_size 8

# ProteinGym quick test (limit max samples)
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m model.pt -d data.csv --max_samples 100
```

#### Data Management Options

```bash
# Specify custom output directory (common to all evaluation scripts)
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m model.pt -d data.csv -o /custom/output/path

./workflows/run_bert_clinvar_evaluation.sh \
  --output_dir /custom/clinvar/results

./workflows/run_gpt2_omim_evaluation_real.sh \
  --output_dir /custom/omim/results

# Specify data prep and report output separately
# (some scripts support --data_dir and --output_dir independently)

# Reuse existing OMIM data (skip download)
./workflows/run_gpt2_omim_evaluation_real.sh \
  --existing_omim_dir /path/to/omim_data

# Auto-create ProteinGym sample data
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m model.pt --create-sample
```

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

Automated model evaluation scripts integrating three phases: data preparation, evaluation, and visualization

**BERT Models:**

- `run_bert_proteingym_evaluation.sh` - BERT ProteinGym evaluation
- `run_bert_clinvar_evaluation.sh` - BERT ClinVar evaluation

**GPT-2 Genome Sequence:**

- `run_gpt2_clinvar_evaluation.sh` - GPT-2 ClinVar evaluation
- `run_gpt2_cosmic_evaluation.sh` - GPT-2 COSMIC evaluation
- `run_gpt2_omim_evaluation_dummy.sh` - GPT-2 OMIM evaluation (sample)
- `run_gpt2_omim_evaluation_real.sh` - GPT-2 OMIM evaluation (real data)

**GPT-2 Protein Sequence:**

- `run_gpt2_proteingym_evaluation.sh` - GPT-2 ProteinGym evaluation
- `run_gpt2_protein_classification.sh` - GPT-2 Protein Classification evaluation

**RNA Sequence:**

- `run_rna_benchmark_evaluation.sh` - RNA Benchmark evaluation

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

All evaluation scripts support customizable output paths:

```bash
# Default output (under LEARNING_SOURCE_DIR)
./workflows/run_bert_proteingym_evaluation.sh
# -> $LEARNING_SOURCE_DIR/protein_sequence/report/bert_proteingym_YYYYMMDD_HHMMSS/

# Specify custom output path
./workflows/run_bert_proteingym_evaluation.sh \
  --output_dir /mnt/results/my_proteingym_eval
# -> /mnt/results/my_proteingym_eval/

# Relative path also supported
./workflows/run_gpt2_clinvar_evaluation.sh \
  -o ./my_clinvar_results
# -> ./my_clinvar_results/

# Specify data prep and report output separately (some scripts)
./workflows/run_gpt2_omim_evaluation_real.sh \
  --output_dir /results/omim_eval \
  --config /custom/config.yaml
```

**Default output paths:**

| Script | Default Output Path |
|--------|---------------------|
| `run_bert_clinvar_evaluation.sh` | `$LEARNING_SOURCE_DIR/genome_sequence/report/bert_clinvar_evaluation` |
| `run_bert_proteingym_evaluation.sh` | `$LEARNING_SOURCE_DIR/protein_sequence/report/bert_proteingym` |
| `run_gpt2_clinvar_evaluation.sh` | `$LEARNING_SOURCE_DIR/genome_sequence/report/clinvar_evaluation` |
| `run_gpt2_cosmic_evaluation.sh` | `$LEARNING_SOURCE_DIR/genome_sequence/report/cosmic_evaluation` |
| `run_gpt2_omim_evaluation_dummy.sh` | `$LEARNING_SOURCE_DIR/genome_sequence/report/omim_evaluation` |
| `run_gpt2_omim_evaluation_real.sh` | `$LEARNING_SOURCE_DIR/genome_sequence/report/omim_real_evaluation` |
| `run_gpt2_proteingym_evaluation.sh` | `$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_proteingym` |
| `run_gpt2_protein_classification.sh` | `$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_protein_classification` |

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
- **Real data access**: `run_gpt2_omim_evaluation_real.sh` requires OMIM authentication
- **Sample data**: `_dummy.sh` scripts require no authentication and are suitable for development/testing

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

- **Default device**: All evaluation scripts use GPU (cuda) by default
- **CPU fallback**: Use `--device cpu` option to run on CPU (slower)
- **Sample limit**: Use `--max_samples N` to speed up test runs
- **Batch size tuning**: Use `--batch_size N` to control memory usage
- **Data reuse**: Use `--existing_omim_dir` to avoid re-downloading

### New Features

- **Protein Classification visualization**: Automatically generates 10+ charts and detailed HTML reports
- **ProteinGym sample data**: Use `--create-sample` to automatically download the recommended dataset
- **OMIM data reuse**: Use `--existing_omim_dir` to leverage already-downloaded data
- **Default model**: Protein Classification can run without specifying a model
- **ClinVar balanced sampling**: Accurate evaluation with 1,000 pathogenic and 1,000 benign variants each

### ClinVar Balanced Sampling Details

#### Background

Previous ClinVar data preparation extracted only a few samples, resulting in low evaluation reliability.

#### Improvements

Using `extract_random_clinvar_samples.py` to achieve the following:

**Dataset composition**:

- Pathogenic variants: 1,000 samples
- Benign variants: 1,000 samples
- Total: 2,000 balanced samples

**Sampling method**:

1. Fetch ClinVar data from HuggingFace Datasets
2. Automatically classify by clinical significance (pathogenic/benign)
3. Randomly sample 1,000 from each class
4. Extract flanking sequences from the reference genome

**Benefits**:

- Eliminates class imbalance for accurate accuracy evaluation
- Reproducible random sampling (fixed seed=42)
- Automated data preparation workflow

**Usage**:

```bash
# GPT-2 ClinVar evaluation
./workflows/run_gpt2_clinvar_evaluation.sh --download

# BERT ClinVar evaluation
./workflows/run_bert_clinvar_evaluation.sh --prepare-data
```

## 📞 Troubleshooting

### Common Issues and Solutions

1. **Environment variable error**

   ```bash
   # Error: LEARNING_SOURCE_DIR environment variable is not set
   export LEARNING_SOURCE_DIR=/path/to/learning_source
   ```

2. **Model file not found**

   ```bash
   # Check model directory
   ls -la gpt2-output/
   ls -la runs_train_bert_*/

   # Protein Classification uses default model automatically
   ./workflows/run_gpt2_protein_classification.sh -s
   # -> uses gpt2-output/protein_sequence-small/ckpt.pt automatically
   ```

3. **CUDA error**

   ```bash
   # Check GPU
   nvidia-smi

   # Switch to CPU (supported by all scripts)
   ./workflows/run_gpt2_*.sh --device cpu

   # Note: CPU is approx. 4x slower than GPU
   ```

4. **Data file not found**

   ```bash
   # Re-run data preparation phase
   ./workflows/run_gpt2_*.sh --force_download

   # Or run data preparation only
   ./workflows/run_gpt2_*.sh --skip_evaluation --skip_visualization

   # Auto-create ProteinGym sample data
   ./workflows/run_gpt2_proteingym_evaluation.sh \
     -m model.pt --create-sample

   # Create ClinVar balanced sampling data
   # For GPT-2
   ./workflows/run_gpt2_clinvar_evaluation.sh --download
   # For BERT
   ./workflows/run_bert_clinvar_evaluation.sh --prepare-data
   ```

5. **OMIM real data access error**

   ```bash
   # Verify authentication URL is correctly configured
   cat assets/configs/omim_real_data.yaml

   # Verify operation with sample data
   ./workflows/run_gpt2_omim_evaluation_dummy.sh

   # Reuse existing data (avoid re-downloading)
   ./workflows/run_gpt2_omim_evaluation_real.sh \
     --existing_omim_dir /path/to/omim_data
   ```

6. **Missing Python packages**

   ```bash
   # Install required packages
   pip install torch transformers pandas numpy scikit-learn matplotlib seaborn sentencepiece scipy
   ```

7. **ProteinGym evaluation is slow**

   ```bash
   # Use GPU (default, approx. 4x faster)
   ./workflows/run_gpt2_proteingym_evaluation.sh -m model.pt -d data.csv

   # Limit sample count for testing
   ./workflows/run_gpt2_proteingym_evaluation.sh \
     -m model.pt -d data.csv --max_samples 100

   # Progress: 50 samples/GPU ~ 12 sec, 2770 samples/GPU ~ 11 min
   ```

8. **Visualization error**

   ```bash
   # Check if evaluation results exist
   ls -la $LEARNING_SOURCE_DIR/*/report/*/

   # Re-run visualization only
   ./workflows/run_gpt2_*.sh --skip_data_prep --skip_evaluation

   # Protein Classification detailed report
   # -> 10+ charts + HTML in visualizations/ directory

   # Specify custom output and re-run visualization
   ./workflows/run_gpt2_proteingym_evaluation.sh \
     --skip_data_prep --skip_evaluation \
     -o /custom/visualization/path
   ```

9. **Output directory not found**

   ```bash
   # Check default output path
   echo $LEARNING_SOURCE_DIR
   ls -la $LEARNING_SOURCE_DIR/*/report/

   # If custom output was used
   ls -la /path/to/custom/output/

   # Re-run with explicit output path
   ./workflows/run_bert_proteingym_evaluation.sh \
     --output_dir /specific/output/path

   # Find the latest evaluation result directory
   find $LEARNING_SOURCE_DIR -type d -name "*proteingym*" -o -name "*clinvar*" | sort
   ```

10. **ClinVar data has only a few samples**

```bash
# Problem: legacy method extracts only a small number of samples
# Solution: use balanced sampling script

# For GPT-2 (auto-generates 2,000 balanced samples)
./workflows/run_gpt2_clinvar_evaluation.sh --download

# For BERT (auto-generates 2,000 balanced samples)
./workflows/run_bert_clinvar_evaluation.sh --prepare-data

# Check dataset statistics
python -c "
import pandas as pd
df = pd.read_csv('$LEARNING_SOURCE_DIR/genome_sequence/data/clinvar/clinvar_evaluation_dataset.csv')
print(f'Total samples: {len(df)}')
print(df['ClinicalSignificance'].value_counts())
"
# Expected: Pathogenic 1,000, Benign 1,000
```

11. **Reference genome file not found (ClinVar balanced sampling)**

    ```bash
    # Download reference genome
    wget -P $LEARNING_SOURCE_DIR/genome_sequence/data/ \
      https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.28_GRCh38.p13/GCA_000001405.28_GRCh38.p13_genomic.fna.gz

    # If already downloaded, verify path
    ls -lh $LEARNING_SOURCE_DIR/genome_sequence/data/GCA_000001405.28_GRCh38.p13_genomic.fna*

    # .gz files can be used as-is (scripts auto-decompress)
    ```

12. **Batch-testing multiple GPT-2 checkpoints**

    ```bash
    # Batch test all domain checkpoints
    ./workflows/batch_test_gpt2.sh gpt2-output/

    # Test only a specific subdirectory
    ./workflows/batch_test_gpt2.sh path/to/checkpoints/

    # Results saved to gpt2_test_results_TIMESTAMP/
    ls -la gpt2_test_results_*/

    # Check per-domain results
    # - compounds: compound generation validity
    # - genome_sequence: genomic sequence consistency
    # - protein_sequence: protein sequence quality
    # - rna: RNA sequence structural validity
    # - molecule_nat_lang: molecular description text quality
    ```

### Log Inspection

Each script outputs detailed logs:

- Console output: Real-time progress
- `logs/`: System logs (some scripts)
- `$OUTPUT_DIR/*_report.txt`: Detailed evaluation result reports

## 🔄 Migration Notes

### Script Structural Changes

These scripts have undergone the following changes:

1. **Clarified file naming**
   - Added `run_gpt2_` prefix to GPT-2-specific scripts
   - Added `run_bert_` prefix to BERT-specific scripts
   - Added `_real` suffix to OMIM real data scripts

2. **Three-phase integration**
   - Merged data preparation, evaluation, and visualization scripts
   - Added per-phase skip options

3. **Unified LEARNING_SOURCE_DIR structure**
   - Consistent directory structure across all scripts
   - Added environment variable checks

4. **Unified script paths**
   - All Python execution paths unified under `molcrawl/tasks/evaluation/{task}/{arch}_*.py (organized by task)`

All functionality is identical as long as scripts are run from the project root directory.
