# Bootstrap Scripts

Project initialization, evaluation, and maintenance scripts for the RIKEN Dataset Foundational Model project.

## 📋 Overview

This directory contains shell scripts for various project operations. All scripts should be executed from the project root directory unless otherwise specified.

```bash
# Usage pattern
cd /path/to/riken-dataset-fundational-model
./bootstraps/script_name.sh
```

## 🚀 AI Model Evaluation Scripts

### Protein Sequence Models
| Script | Purpose | Model Type | Dataset | Output Location |
|--------|---------|------------|---------|----------------|
| `run_bert_proteingym_evaluation.sh` | BERT protein fitness prediction | BERT | ProteinGym | `$LEARNING_SOURCE_DIR/protein_sequence/report/bert_proteingym_*` |
| `run_proteingym_evaluation.sh` | GPT-2 protein fitness prediction | GPT-2 | ProteinGym | `$LEARNING_SOURCE_DIR/protein_sequence/report/proteingym_*` |
| `run_protein_classification_evaluation.sh` | General protein classification | BERT/GPT-2 | Custom datasets | `$LEARNING_SOURCE_DIR/protein_sequence/report/protein_classification_*` |

### Genome Sequence Models  
| Script | Purpose | Model Type | Dataset | Output Location |
|--------|---------|------------|---------|----------------|
| `run_bert_clinvar_evaluation.sh` | Variant pathogenicity prediction | BERT | ClinVar | `$LEARNING_SOURCE_DIR/genome_sequence/report/clinvar_*` |
| `run_omim_real_evaluation.sh` | Disease variant analysis | GPT-2 | OMIM | `$LEARNING_SOURCE_DIR/genome_sequence/report/omim_*` |

### Data Processing & Visualization
| Script | Purpose | Function | Dependencies |
|--------|---------|----------|-------------|
| `run_bert_proteingym_data_prep.sh` | ProteinGym data preprocessing | Data pipeline setup | Raw ProteinGym datasets |
| `run_bert_proteingym_visualization.sh` | Generate evaluation plots | Result visualization | Completed evaluations |

## 🔧 Development & Debugging

### System Debugging
| Script | Purpose | Use Case |
|--------|---------|----------|
| `debug_protein_bert.sh` | BERT protein model debugging | Troubleshooting training issues |
| `reboot-cause-check.sh` | System reboot analysis | Infrastructure monitoring |
| `test_bert_checkpoint.sh` | BERT checkpoint validation | Model testing |

### Development Utilities
| Script | Purpose | Function |
|--------|---------|----------|
| `create_sample_vocab.sh` | Generate sample vocabulary files | Development setup |

## 🏗️ Experiment Management System

### Infrastructure Scripts
| Script | Purpose | Function | Port/Service |
|--------|---------|----------|-------------|
| `setup_experiment_system.sh` | Initialize experiment tracking | System configuration | - |
| `start_experiment_system.sh` | Launch experiment services | Service orchestration | Multiple services |
| `demo_experiment_system.sh` | System demonstration | Testing & validation | Demo mode |
| `start_api_server.py` | Web API for experiments | RESTful service | Default: 8000 |

## 📊 Output Structure

All evaluation scripts use the structured output format:

```
$LEARNING_SOURCE_DIR/
├── genome_sequence/
│   └── report/
│       ├── clinvar_bert_YYYYMMDD_HHMMSS/
│       ├── clinvar_gpt2_YYYYMMDD_HHMMSS/
│       └── omim_gpt2_YYYYMMDD_HHMMSS/
└── protein_sequence/
    └── report/
        ├── bert_proteingym_YYYYMMDD_HHMMSS/
        ├── proteingym_gpt2_YYYYMMDD_HHMMSS/
        └── protein_classification_YYYYMMDD_HHMMSS/
```

Each evaluation directory contains:
- `evaluation_results.json` - Structured results
- `evaluation_report.txt` - Human-readable summary  
- `evaluation_plots.png` - Visualization charts
- `detailed_results.csv` - Per-sample predictions

## 🎯 Quick Start Examples

### Standard Evaluations
```bash
# BERT ProteinGym evaluation
./bootstraps/run_bert_proteingym_evaluation.sh --dataset data/proteingym/sample.csv

# BERT ProteinGym evaluation with balanced sampling (1000 positive + 1000 negative)
./bootstraps/run_bert_proteingym_evaluation.sh --dataset data/proteingym/sample.csv --balanced

# GPT-2 ProteinGym evaluation  
./bootstraps/run_proteingym_evaluation.sh --dataset data/proteingym/sample.csv

# ClinVar variant analysis
./bootstraps/run_bert_clinvar_evaluation.sh
```

### Balanced Dataset Preparation
```bash
# Prepare balanced ProteinGym data (1000 positive + 1000 negative samples)
python scripts/proteingym_data_preparation.py \
  --prepare_assay BLAT_ECOLX_Ranganathan2015 \
  --balanced_sampling \
  --positive_samples 1000 \
  --negative_samples 1000

# Prepare multiple assays with balanced sampling
python scripts/proteingym_data_preparation.py \
  --prepare_multiple_assays BLAT_ECOLX_Ranganathan2015 CALM1_HUMAN_Roth2017 \
  --positive_samples 500 \
  --negative_samples 500 \
  --output_dir ./balanced_proteingym_data

# Create balanced sample data for testing
python scripts/proteingym_evaluation.py \
  --model_path runs_train_bert_protein_sequence/checkpoint-5000 \
  --proteingym_data sample_balanced.csv \
  --create_sample_data \
  --balanced_samples \
  --sample_positive_count 1000 \
  --sample_negative_count 1000
```

### Experiment System
```bash
# Complete system setup
./bootstraps/setup_experiment_system.sh

# Start all services
./bootstraps/start_experiment_system.sh

# Demo the system
./bootstraps/demo_experiment_system.sh
```

### Development Workflow
```bash
# Debug BERT training
./bootstraps/debug_protein_bert.sh

# Test model checkpoints
./bootstraps/test_bert_checkpoint.sh

# Create development vocabularies
./bootstraps/create_sample_vocab.sh
```

## 🔧 Prerequisites

### Environment Setup
```bash
# Required environment variables
export LEARNING_SOURCE_DIR=/path/to/learning_source_202508
export CUDA_VISIBLE_DEVICES=0  # For GPU usage

# Load project configuration
source src/config/env.sh
```

### Dependencies
- Python 3.8+ with transformers, torch, pandas, numpy
- CUDA-capable GPU for model training/evaluation
- Sufficient disk space for datasets and results
- Access to model checkpoints in appropriate directories

## 📝 Script Categories

### 🔍 **Evaluation Scripts** (5 scripts)
Automated evaluation of trained models on various datasets with standardized output formatting.

### 🛠️ **Development Scripts** (4 scripts)  
Debugging, testing, and development utilities for model development workflow.

### 🏭 **Infrastructure Scripts** (4 scripts)
System setup, service management, and experiment tracking infrastructure.

### ⚙️ **Utility Scripts** (1 script)
Helper scripts for data preparation and project setup tasks.

## ⚖️ Balanced Sampling Features

### Dataset Balance
- **Default Configuration**: 1000 positive + 1000 negative samples
- **Automatic Threshold**: Uses median DMS score if no threshold specified
- **Flexible Sampling**: Customizable positive/negative sample counts
- **Quality Assurance**: Maintains data distribution while ensuring balance

### Benefits
- **Reduced Bias**: Eliminates class imbalance issues in evaluation
- **Consistent Evaluation**: Standardized dataset sizes across experiments
- **Fair Comparison**: Enables meaningful model performance comparisons
- **Resource Efficiency**: Fixed dataset sizes for predictable resource usage

## 🚨 Important Notes

- **Execution Location**: All scripts must be run from project root directory
- **Output Management**: Results are automatically timestamped and organized
- **Resource Management**: GPU memory and disk space requirements vary by script
- **Logging**: Comprehensive logging is provided for all operations
- **Error Handling**: Scripts include robust error checking and recovery
- **Balanced Sampling**: New default for ProteinGym evaluations ensures fair model comparison

## 📞 Troubleshooting

For common issues:
1. Check `LEARNING_SOURCE_DIR` environment variable is set
2. Ensure CUDA drivers are properly installed for GPU scripts
3. Verify model checkpoints exist in expected locations
4. Review logs in respective output directories for detailed error messages

## 🔄 Migration Notes

These scripts have been moved from the project root to improve organization. All functionality remains identical when executed from the project root directory.