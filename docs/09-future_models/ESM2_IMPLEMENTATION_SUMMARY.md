# ESM-2 Implementation Summary

## Created on: 2026-01-22

## Completed Components

### 1. Core Files

- [esm2/main.py](../../molcrawl/models/esm2/main.py) - Main training script (465 lines)
- [esm2/configurator.py](../../molcrawl/models/esm2/configurator.py) - Configuration loader
- [esm2/configs/protein_sequence.py](../../molcrawl/tasks/pretrain/configs/protein_sequence/esm2.py) - Protein-sequence config

### 2. Workflow Scripts

- [workflows/03e-protein_sequence-train-esm2-small.sh](../../workflows/03e-protein_sequence-train-esm2-small.sh)
- [workflows/03e-protein_sequence-train-esm2-medium.sh](../../workflows/03e-protein_sequence-train-esm2-medium.sh)
- [workflows/03e-protein_sequence-train-esm2-large.sh](../../workflows/03e-protein_sequence-train-esm2-large.sh)

### 3. Documentation

- [docs/ESM2_TRAINING_GUIDE.md](ESM2_TRAINING_GUIDE.md) - Detailed guide

## Key Characteristics

### ESM-2 vs Existing BERT Setup

| Feature      | Existing BERT | ESM-2              |
| ------------ | ------------- | ------------------ |
| Domain       | General       | Protein-specific   |
| Tokenization | Character     | Amino-acid level   |
| Learning Rate| 6e-6          | 4e-4               |
| Dropout      | 0.1           | 0.0                |
| Optimization | Standard      | Protein-optimized  |
| Convergence  | Normal        | Faster             |

### Model Sizes

- **Small** (320d, 6 layers): ~8M params, development/testing
- **Medium** (480d, 12 layers): ~35M params, experiments
- **Large** (640d, 30 layers): ~150M params, production-oriented runs

## Quick Start

```bash
# Start small model training
cd <PROJECT_ROOT>
CUDA_VISIBLE_DEVICES=0 ./workflows/03e-protein_sequence-train-esm2-small.sh
```

```bash
# Enable Weights & Biases
CUDA_VISIBLE_DEVICES=0 USE_WANDB=True WANDB_PROJECT=esm2-protein \
  ./workflows/03e-protein_sequence-train-esm2-small.sh
```

```bash
# Check logs
tail -f $LEARNING_SOURCE_DIR/protein_sequence/logs/esm2-train-small-*.log
ls -lt $LEARNING_SOURCE_DIR/protein_sequence/logs/esm2-train-*.log | head -1
```

## Dataset

- **Source**: `protein_sequence/training_ready_hf_dataset/`
- **Content**: UniProt UniRef50 protein sequences
- **Tokenizer**: ESM character-level tokenizer (BERT-compatible usage)
- **Additional preparation**: Not required

## Configuration Customization

```bash
python esm2/main.py esm2/configs/protein_sequence.py \
  --model_size=medium \
  --max_steps=600000 \
  --learning_rate=5e-4 \
  --batch_size=8 \
  --save_steps=10000
```

Edit [esm2/configs/protein_sequence.py](../../molcrawl/tasks/pretrain/configs/protein_sequence/esm2.py):

```python
batch_size = 8                   # default: 4
gradient_accumulation_steps = 16 # default: 32
save_steps = 2000                # default: 5000
```
