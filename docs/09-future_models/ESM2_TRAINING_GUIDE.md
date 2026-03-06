# ESM-2 Training Guide for Protein Sequence Data

## Overview

This guide explains how to train ESM-2 using the existing `protein_sequence` dataset (UniProt).

## What is ESM-2?

ESM-2 (Evolutionary Scale Modeling 2) is a state-of-the-art Transformer model specialized for protein sequences.

### Key Points

1. **Evolution-scale pretraining**
   - Pretrained on large protein corpora
   - Learns biologically meaningful sequence representations

2. **Scalable architecture**
   - Model families range from small to very large scales
   - Efficient for transfer to downstream protein tasks

3. **Broad downstream applicability**
   - Structure-related tasks
   - Function annotation
   - Variant effect prediction
   - Contact-related signals

### Comparison with Existing BERT Setup

| Item             | BERT (existing) | ESM-2              |
| ---------------- | --------------- | ------------------ |
| Domain           | General         | Protein-specific   |
| Tokenization     | Character       | Amino-acid level   |
| Learning Rate    | 6e-6            | 4e-4               |
| Dropout          | 0.1             | 0.0                |
| Position Embeds  | Learned         | Learned            |
| Optimizer        | AdamW           | AdamW              |
| Convergence      | Normal          | Faster             |

## Setup

### 1. Dependencies

No extra package is required beyond the current environment.
(`transformers`, `datasets` are already used.)

### 2. Directory Layout

```text
esm2/
├── main.py                     # Main training script
├── configurator.py             # Config loader
└── configs/
    └── protein_sequence.py     # protein_sequence config

workflows/
├── 03e-protein_sequence-train-esm2-small.sh
├── 03e-protein_sequence-train-esm2-medium.sh
└── 03e-protein_sequence-train-esm2-large.sh
```

## Usage

```bash
# Small (recommended for dev/test)
CUDA_VISIBLE_DEVICES=0 ./workflows/03e-protein_sequence-train-esm2-small.sh

# With W&B
CUDA_VISIBLE_DEVICES=0 USE_WANDB=True WANDB_PROJECT=esm2-protein \
  ./workflows/03e-protein_sequence-train-esm2-small.sh

# Medium
CUDA_VISIBLE_DEVICES=0,1 ./workflows/03e-protein_sequence-train-esm2-medium.sh

# Large
CUDA_VISIBLE_DEVICES=0,1,2,3 ./workflows/03e-protein_sequence-train-esm2-large.sh
```

```bash
# Tail logs
tail -f $LEARNING_SOURCE_DIR/protein_sequence/logs/esm2-train-small-*.log

# Latest log file
ls -lt $LEARNING_SOURCE_DIR/protein_sequence/logs/esm2-train-*.log | head -1

# Find/stop process
ps aux | grep esm2
kill <PID>
```

## Model Size Guidance

### Small (~8M)

- Hidden size: 320
- Layers: 6
- Attention heads: 20
- GPU memory: ~8GB (batch size 4)
- Best for environment validation and iteration

### Medium (~35M)

- Better quality/compute trade-off for most experiments

### Large (~150M)

- Multi-GPU training recommended
- Better suited for production-scale targets
