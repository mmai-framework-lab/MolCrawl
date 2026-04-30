# ChemBERTa-2 Training Guide

## Overview

ChemBERTa-2 is a RoBERTa-based Transformer specialized for SMILES compound data.
It is pre-trained on large compound corpora (for example, Organix13) and can be adapted to downstream tasks such as molecular property prediction and molecule generation.

## Features

### Domain Focus: SMILES Compounds

- SMILES-specific tokenization
- Strong transfer learning behavior on molecular property tasks
- Improved representation learning for molecular structure patterns

### Technical Specifications

| Model Size | Parameters | Hidden Size | Layers | Attention Heads | Intermediate Size |
| ---------- | ---------- | ----------- | ------ | --------------- | ----------------- |
| Small      | ~10M       | 384         | 6      | 6               | 1536              |
| Medium     | ~85M       | 768         | 12     | 12              | 3072              |
| Large      | ~355M      | 1024        | 24     | 16              | 4096              |

### Training Configuration

- **Architecture**: RoBERTa
- **Learning Rate**: `6e-5` (optimized for SMILES data)
- **Batch Size**: 128 per device
- **Gradient Accumulation**: 1 (effective batch size 128)
- **Max Sequence Length**: 256
- **Mixed Precision**: FP16
- **Optimizer**: AdamW + linear schedule
- **Warmup Steps**: 10,000

## Quick Start

### 1. Check dataset location

```bash
ls -la learning_source
```

### 2. Run training

```bash
# Small
CUDA_VISIBLE_DEVICES=0 ./workflows/03g-compounds-train-chemberta2-small.sh

# Medium
CUDA_VISIBLE_DEVICES=0,1 ./workflows/03g-compounds-train-chemberta2-medium.sh

# Large
CUDA_VISIBLE_DEVICES=0,1,2,3 ./workflows/03g-compounds-train-chemberta2-large.sh
```

### 3. Enable Weights & Biases

```bash
LEARNING_SOURCE_DIR=learning_source \
USE_WANDB=True \
WANDB_PROJECT=chemberta2-compounds \
  ./workflows/03g-compounds-train-chemberta2-small.sh
```

## Dataset Format

- **Input**: SMILES strings (example: `CC(C)Cc1ccc(cc1)C(C)C(=O)O`)
- **Vocabulary**: ~612 tokens (SMILES symbols + special tokens)
- **Sequence Length**: 256 (typical SMILES length: 50-150)
- **Dataset Size**: ~10M compounds (Organix13)

## Model Architecture

1. **Input Embedding**: Convert SMILES tokens to vectors
2. **RoBERTa Encoder**: Multi-layer self-attention stack
3. **MLM Objective**: Predict masked tokens

### RoBERTa vs BERT (relevant differences)

- Dynamic masking
- No NSP objective
- Larger effective batches
- Typically trained on larger corpora

## Directory Structure

```text
molcrawl/models/chemberta2/
├── main.py                      # Main training script
├── configurator.py              # Config loader
└── configs/
    └── compounds.py             # Compounds config

workflows/
├── 03g-compounds-train-chemberta2-small.sh
├── 03g-compounds-train-chemberta2-medium.sh
└── 03g-compounds-train-chemberta2-large.sh

learning_source
└── compounds/
    ├── organix13/
    │   └── compounds/
    │       └── training_ready_hf_dataset/  # Training dataset
    └── chemberta2-output/                  # Model output
```
