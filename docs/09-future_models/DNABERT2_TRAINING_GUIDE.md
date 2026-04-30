# DNABERT-2 Training Guide for Genome Sequence Data

## Overview

This guide describes how to train DNABERT-2 using the existing `genome_sequence` dataset.

## What is DNABERT-2?

DNABERT-2 is a BERT-based model specialized for DNA sequence analysis.

### Key Points

1. **BPE Tokenization**
   - Uses BPE instead of traditional fixed-length k-mers
   - More flexible and often more efficient tokenization

2. **Optimized Architecture**
   - Designed for DNA sequence characteristics
   - Better efficiency/quality trade-off

3. **Efficient Training**
   - Faster convergence than the baseline BERT setup
   - Better use of compute resources

### Comparison with Existing BERT Setup

| Item            | BERT (existing) | DNABERT-2        |
| --------------- | --------------- | ---------------- |
| Tokenization    | k-mer           | BPE              |
| Max Length      | 1024            | 512 (efficient)  |
| Learning Rate   | 6e-6            | 3e-5             |
| Batch Size      | 8               | 16               |
| MLM Probability | 0.2             | 0.15             |
| Convergence     | Slower          | Faster           |

## Setup

### 1. Dependencies

No additional packages are required beyond the current environment.
(`transformers`, `datasets`, and `sentencepiece` are already used.)

### 2. Directory Layout

```text
molcrawl/models/dnabert2/
├── main.py                     # Main training script
├── configurator.py             # Config loader
└── configs/
    └── genome_sequence.py      # genome_sequence config

workflows/
├── 03d-genome_sequence-train-dnabert2-small.sh
├── 03d-genome_sequence-train-dnabert2-medium.sh
└── 03d-genome_sequence-train-dnabert2-large.sh
```

## Usage

### Basic training commands

```bash
# Small (recommended for development/testing)
CUDA_VISIBLE_DEVICES=0 ./workflows/03d-genome_sequence-train-dnabert2-small.sh

# Enable W&B logging
CUDA_VISIBLE_DEVICES=0 USE_WANDB=True WANDB_PROJECT=dnabert2-genome \
  ./workflows/03d-genome_sequence-train-dnabert2-small.sh

# Medium (recommended for experiments)
CUDA_VISIBLE_DEVICES=0,1 ./workflows/03d-genome_sequence-train-dnabert2-medium.sh

# Large (recommended for production-scale runs)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./workflows/03d-genome_sequence-train-dnabert2-large.sh
```

### Logging and process control

```bash
# Tail logs
tail -f $LEARNING_SOURCE_DIR/genome_sequence/logs/dnabert2-train-small-*.log

# Latest log file
ls -lt $LEARNING_SOURCE_DIR/genome_sequence/logs/dnabert2-train-*.log | head -1

# Find/stop process
ps aux | grep dnabert2
kill <PID>
```

## Model Size Guidance

### Small (768 dim, 12 layers)

- ~110M parameters
- ~8GB GPU memory (batch size 16)
- Best for development, test, and prototyping

### Medium (1024 dim, 24 layers)

- ~350M parameters
- ~16GB GPU memory (batch size 16)
- Good default for serious experiments

### Large (1280 dim, 32 layers)

- ~600M parameters
- Multi-GPU strongly recommended
- For production-scale quality targets
