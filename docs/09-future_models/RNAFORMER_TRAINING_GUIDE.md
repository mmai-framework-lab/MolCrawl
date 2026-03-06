# RNAformer Training Guide

## Overview

RNAformer is a Transformer model specialized for RNA transcriptome (gene expression) data.
It follows a Geneformer-style architecture and is intended for large-scale single-cell RNA-seq training workflows.

## Features

### Domain Focus: RNA Transcriptome

- Custom tokenization for gene-expression data
- Cell-type-aware representation learning
- Long-context support (up to 1024 tokens)

### Technical Specifications

| Model Size | Parameters | Hidden Size | Layers | Attention Heads | Intermediate Size |
| ---------- | ---------- | ----------- | ------ | --------------- | ----------------- |
| Small      | ~40M       | 512         | 8      | 8               | 2048              |
| Medium     | ~90M       | 768         | 12     | 12              | 3072              |
| Large      | ~180M      | 1024        | 16     | 16              | 4096              |

### Training Configuration

- **Learning Rate**: `1e-4` (optimized for RNA transcriptome)
- **Batch Size**: 8 per device
- **Gradient Accumulation**: 16 (effective batch 128)
- **Max Sequence Length**: 1024
- **Mixed Precision**: FP16
- **Optimizer**: AdamW + cosine schedule
- **Warmup Steps**: 10,000

## Quick Start

### 1. Check dataset location

```bash
ls -la learning_source
```

### 2. Run training

```bash
# Small
CUDA_VISIBLE_DEVICES=0 ./workflows/03f-rna-train-rnaformer-small.sh

# Medium
CUDA_VISIBLE_DEVICES=0 ./workflows/03f-rna-train-rnaformer-medium.sh

# Large
CUDA_VISIBLE_DEVICES=0 ./workflows/03f-rna-train-rnaformer-large.sh
```

### 3. Enable Weights & Biases

```bash
LEARNING_SOURCE_DIR=learning_source \
USE_WANDB=True \
WANDB_PROJECT=rnaformer-transcriptome \
  ./workflows/03f-rna-train-rnaformer-small.sh
```

## Dataset Format

- **Input**: Gene ID sequences (typically sorted by expression)
- **Vocabulary**: ~60,000 genes
- **Sequence Length**: 1024
- **Dataset Scale**: ~54M cells (depending on source snapshot)

## Model Architecture

1. **Input Embedding**: Map gene IDs to vector space
2. **Transformer Encoder**: Multi-layer self-attention
3. **MLM Objective**: Predict masked gene tokens

## Directory Structure

```text
rnaformer/
├── main.py                      # Main training script
├── configurator.py              # Config loader
└── configs/
    └── rna.py                   # RNA transcriptome config

workflows/
├── 03f-rna-train-rnaformer-small.sh
├── 03f-rna-train-rnaformer-medium.sh
└── 03f-rna-train-rnaformer-large.sh

learning_source
└── rna/
    ├── gene_vocab.json              # Gene vocabulary
    ├── training_ready_hf_dataset/   # Training dataset
    └── rnaformer-output/            # Model output
```
