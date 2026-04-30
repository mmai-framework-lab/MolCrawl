# DNABERT-2 Implementation Summary

## Created on: 2026-01-22

## Completed Components

### 1. Core Files

- [dnabert2/main.py](../../molcrawl/models/dnabert2/main.py) - Main training script (433 lines)
- [dnabert2/configurator.py](../../molcrawl/models/dnabert2/configurator.py) - Configuration loader
- [dnabert2/configs/genome_sequence.py](../../molcrawl/models/dnabert2/configs/genome_sequence.py) - Genome-sequence config

### 2. Workflow Scripts

- [workflows/03d-genome_sequence-train-dnabert2-small.sh](../../workflows/03d-genome_sequence-train-dnabert2-small.sh)
- [workflows/03d-genome_sequence-train-dnabert2-medium.sh](../../workflows/03d-genome_sequence-train-dnabert2-medium.sh)
- [workflows/03d-genome_sequence-train-dnabert2-large.sh](../../workflows/03d-genome_sequence-train-dnabert2-large.sh)

### 3. Documentation

- [docs/DNABERT2_TRAINING_GUIDE.md](DNABERT2_TRAINING_GUIDE.md) - Detailed guide

## Key Characteristics

### DNABERT-2 vs Existing BERT Setup

| Feature        | Existing BERT | DNABERT-2        |
| -------------- | ------------- | ---------------- |
| Tokenization   | k-mer         | BPE              |
| Max Length     | 1024          | 512 (efficient)  |
| Learning Rate  | 6e-6          | 3e-5             |
| Batch Size     | 8             | 16               |
| Convergence    | Slower        | Faster           |
| GPU Efficiency | Lower         | Higher           |

### Model Sizes

- **Small** (768d, 12 layers): ~110M params, development/testing
- **Medium** (1024d, 24 layers): ~350M params, experimental runs
- **Large** (1280d, 32 layers): ~600M params, production-oriented runs

## Quick Start

```bash
# Start small model training
cd <PROJECT_ROOT>
CUDA_VISIBLE_DEVICES=0 ./workflows/03d-genome_sequence-train-dnabert2-small.sh
```

```bash
# Enable Weights & Biases
CUDA_VISIBLE_DEVICES=0 USE_WANDB=True WANDB_PROJECT=dnabert2-genome \
  ./workflows/03d-genome_sequence-train-dnabert2-small.sh
```

```bash
# Check logs
tail -f $LEARNING_SOURCE_DIR/genome_sequence/logs/dnabert2-train-small-*.log
ls -lt $LEARNING_SOURCE_DIR/genome_sequence/logs/dnabert2-train-*.log | head -1
```

## Dataset

- **Source**: `genome_sequence/training_ready_hf_dataset/`
- **Content**: RefSeq genome sequences
- **Tokenizer**: Existing SentencePiece tokenizer can be reused
- **Additional preparation**: Not required

## Configuration Customization

```bash
python dnabert2/main.py dnabert2/configs/genome_sequence.py \
  --model_size=medium \
  --max_steps=300000 \
  --learning_rate=5e-5 \
  --batch_size=32 \
  --save_steps=10000
```

Edit [dnabert2/configs/genome_sequence.py](../../molcrawl/models/dnabert2/configs/genome_sequence.py):

```python
max_length = 1024  # default: 512
save_steps = 2000  # default: 5000
```

## Technical Notes

- Mixed precision (FP16)
- Gradient accumulation
- Automatic checkpoint resume
- Multi-worker data loading
- Efficient periodic evaluation
