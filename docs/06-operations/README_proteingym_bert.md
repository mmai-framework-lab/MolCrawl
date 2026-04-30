# ProteinGym Evaluation System for BERT Models

This document explains how to evaluate protein-sequence BERT models with the ProteinGym dataset.

## Overview

The BERT ProteinGym evaluation system uses a trained BERT model to predict fitness effects of protein variants and compares predictions with experimental DMS (Deep Mutational Scanning) scores.

## Key Features

- **Independent implementation**: Fully separated from the GPT-2 evaluation path
- **BERT MLM scoring**: Bidirectional context modeling with masked language modeling
- **EsmSequenceTokenizer**: Protein-sequence-specific tokenizer support
- **Safetensors support**: Efficient model loading
- **Comprehensive metrics**: Correlation metrics and error metrics (MAE, RMSE, etc.)

## File Structure

```text
molcrawl/evaluation/bert/
├── proteingym_evaluation.py        # Main evaluation script
└── proteingym_data_preparation.py  # Dataset download and preprocessing

molcrawl/models/bert/configs/
└── bert_proteingym_config.py       # Configuration

workflows/
└── run_bert_proteingym_evaluation.sh  # Runner script (run from repo root)
```

## Usage

### 1. Basic evaluation run

```bash
# Evaluate on a ProteinGym dataset
./workflows/run_bert_proteingym_evaluation.sh --dataset /path/to/proteingym_data.csv

# Limit sample size
./workflows/run_bert_proteingym_evaluation.sh --dataset /path/to/proteingym_data.csv --sample_size 100
```

### 2. Run with custom options

```bash
# Specify model path and batch size
./workflows/run_bert_proteingym_evaluation.sh \
    --dataset /path/to/proteingym_data.csv \
    --model_path runs_train_bert_protein_sequence/checkpoint-1000 \
    --batch_size 8 \
    --device cuda

# Specify output directory
./workflows/run_bert_proteingym_evaluation.sh \
    --dataset /path/to/proteingym_data.csv \
    --output_dir ./custom_results
```

### 3. Create sample data for testing

```bash
# Create sample input data
./workflows/run_bert_proteingym_evaluation.sh --create_sample_data --dataset ./sample_data.csv

# Evaluate using generated sample data
./workflows/run_bert_proteingym_evaluation.sh --dataset ./sample_data.csv
```

## Prerequisites

### Model

- Trained BERT model: `runs_train_bert_protein_sequence/checkpoint-*`
- Safetensors or PyTorch checkpoint format

### Dataset format

ProteinGym input must include the following columns:

- `mutated_sequence`: Mutated protein sequence
- `DMS_score`: Experimental fitness score
- `target_seq`: Wild-type sequence (optional)
- `mutant`: Mutation descriptor (optional)

### Environment

- CUDA-capable GPU (recommended)
- Conda environment: `conda activate riken-fm`
- Required packages: `torch`, `transformers`, `pandas`, `numpy`, `scipy`, `safetensors`

## Outputs

After evaluation, the following files are generated:

```text
bert_proteingym_evaluation_results/
├── bert_proteingym_results.json              # Main results (JSON)
├── bert_proteingym_detailed_results.csv      # Detailed predictions (CSV)
└── bert_proteingym_evaluation_report.txt     # Text report
```

### Main metrics

- **Spearman correlation**: Rank correlation (ranking quality)
- **Pearson correlation**: Linear correlation
- **MAE**: Mean absolute error
- **RMSE**: Root mean squared error

### BERT-specific analysis

- **MLM score**: Sequence likelihood proxy from masked language modeling
- **Fitness score**: Mutant MLM score minus wild-type MLM score
- **Sequence similarity**: Cosine similarity of `[CLS]` representations
- **Pathogenicity score**: Composite score combining fitness and similarity

## Example Run

```bash
# Activate conda environment
conda activate riken-fm

# Run with sample data
./workflows/run_bert_proteingym_evaluation.sh \
    --create_sample_data \
    --dataset ./sample_data.csv \
    --visualize
```
