# ChemBERTa-2 Implementation Summary

## 📁 Files Created

### Core Implementation

1. **chemberta2/main.py** (370 lines)
   - Main training script with CompoundsDatasetLoader class
   - RoBERTa-based architecture
   - Supports checkpoint resumption
   - Weights & Biases integration
   - Mixed precision training (FP16)

2. **chemberta2/configurator.py** (53 lines)
   - Command-line argument parsing
   - Configuration file loading with overrides

3. **chemberta2/configs/compounds.py** (115 lines)
   - SMILES compounds dataset configuration
   - SMILES tokenizer setup (612 tokens)
   - Preprocessing functions
   - Organix13 dataset integration

### Bootstrap Scripts

1. **workflows/03g-compounds-train-chemberta2-small.sh**
2. **workflows/03g-compounds-train-chemberta2-medium.sh**
3. **workflows/03g-compounds-train-chemberta2-large.sh**
   - Executable training scripts for 3 model sizes
   - Environment variable configuration
   - Automatic logging

### Documentation

1. **docs/CHEMBERTA2_TRAINING_GUIDE.md**
   - Comprehensive user guide
   - Quick start examples
   - Troubleshooting tips
   - Performance benchmarks

## 🚀 Quick Start

### Training Commands

```bash
# Small model (recommended for testing)
CUDA_VISIBLE_DEVICES=0 ./workflows/03g-compounds-train-chemberta2-small.sh

# Medium model
CUDA_VISIBLE_DEVICES=0 ./workflows/03g-compounds-train-chemberta2-medium.sh

# Large model
CUDA_VISIBLE_DEVICES=0,1,2,3 ./workflows/03g-compounds-train-chemberta2-large.sh

# With Weights & Biases
LEARNING_SOURCE_DIR=learning_source_20251210 \
USE_WANDB=True \
WANDB_PROJECT=chemberta2-compounds \
  ./workflows/03g-compounds-train-chemberta2-small.sh
```

## 🔧 Model Specifications

| Model Size | Parameters | Hidden | Layers | Heads | Intermediate |
| ---------- | ---------- | ------ | ------ | ----- | ------------ |
| Small      | ~10M       | 384    | 6      | 6     | 1536         |
| Medium     | ~85M       | 768    | 12     | 12    | 3072         |
| Large      | ~355M      | 1024   | 24     | 16    | 4096         |

## ⚙️ Key Features

### SMILES Compounds Specific

- **Tokenization**: SMILES character-based (612 tokens)
- **Max Length**: 256 tokens (optimal for SMILES)
- **Architecture**: RoBERTa (improved BERT)
- **Dataset**: Organix13 (~10M compounds)

### Training Optimizations

- **Learning Rate**: 6e-5 (optimized for SMILES)
- **Batch Size**: 128 per device (large batches)
- **Gradient Accumulation**: 1 step (effective batch = 128)
- **Mixed Precision**: FP16 for faster training
- **Warmup**: 10,000 steps with linear schedule

### RoBERTa Improvements over BERT

- Dynamic masking (different each epoch)
- No Next Sentence Prediction (NSP) task
- Larger batches and more training data
- Better performance on downstream tasks

## 📊 Comparison with Other Implementations

| Feature       | ChemBERTa-2      | DNABERT-2     | ESM-2             | RNAformer         |
| ------------- | ---------------- | ------------- | ----------------- | ----------------- |
| Domain        | SMILES compounds | DNA sequences | Protein sequences | RNA transcriptome |
| Architecture  | RoBERTa          | BERT          | ESM (BERT-like)   | BERT              |
| Tokenization  | SMILES chars     | BPE           | Amino acids       | Gene IDs          |
| Vocab Size    | 612              | ~4K           | ~33               | ~60K              |
| Max Length    | 256              | 1024          | 1024              | 1024              |
| Learning Rate | 6e-5             | 3e-5          | 4e-4              | 1e-4              |
| Batch Size    | 128              | 16            | 4                 | 8                 |
| Dropout       | 0.1              | 0.1           | 0.0               | 0.1               |

## 🎯 Use Cases

1. **Molecular Property Prediction**: Toxicity, solubility, bioactivity
2. **Drug Discovery**: Lead optimization, ADMET prediction
3. **Retrosynthesis**: Synthesis route prediction
4. **Molecule Generation**: Novel compound design
5. **Reaction Prediction**: Chemical reaction outcome prediction

## 📈 Expected Performance

### Training Metrics

- **Loss**: Should decrease to ~1.5-2.0 after 300K steps
- **Perplexity**: Target ~5-8 for good convergence
- **Memory Usage**: 6-40 GB depending on model size

### Computational Requirements

- **Small**: ~60 hours on A100 (300K steps)
- **Medium**: ~120 hours on A100 (300K steps)
- **Large**: ~300 hours on A100 (300K steps)

## 🛠️ Technical Details

### Data Flow

1. Load SMILES vocabulary (612 tokens)
2. Load Organix13 compounds dataset
3. Add attention masks
4. Apply MLM with 15% masking
5. Train with AdamW optimizer

### Directory Structure

```text
learning_source_20251210/
└── compounds/
    ├── organix13/
    │   └── compounds/
    │       └── training_ready_hf_dataset/  # ~10M compounds
    │           ├── train/
    │           ├── valid/
    │           └── test/
    ├── chemberta2-output/            # Checkpoints
    │   ├── chemberta2-small/
    │   ├── chemberta2-medium/
    │   └── chemberta2-large/
    └── logs/                        # Training logs
```

## 🔍 Validation

### Pre-training Checks

```bash
# 1. Verify dataset
python -c "from datasets import load_from_disk; \
  ds = load_from_disk('learning_source_20251210/compounds/organix13/compounds/training_ready_hf_dataset/train'); \
  print(f'Dataset size: {len(ds)}')"

# 2. Check vocabulary
python -c "with open('assets/molecules/vocab.txt') as f: \
  print(f'Vocab size: {len(f.readlines())}')"

# 3. Test tokenizer
python -c "from compounds.utils.tokenizer import CompoundsTokenizer; \
  tok = CompoundsTokenizer('assets/molecules/vocab.txt', 256); \
  print(f'Tokenizer loaded: {len(tok)} tokens')"
```

### Post-training Checks

```bash
# Check model output
ls -la learning_source_20251210/compounds/chemberta2-output/chemberta2-small/

# View logs
tail -f learning_source_20251210/compounds/logs/chemberta2-train-small-*.log
```

## 🐛 Common Issues

### Issue: Vocabulary file not found

```bash
# Solution: Verify assets directory
ls -la assets/molecules/vocab.txt
```

### Issue: OOM (Out of Memory)

```bash
# Solution: Reduce batch size
python chemberta2/main.py --config chemberta2/configs/compounds.py --batch_size 64
```

### Issue: Training too slow

```bash
# Solution: Use gradient accumulation if batch size is reduced
python chemberta2/main.py --batch_size 64 --gradient_accumulation_steps 2
```

## 📚 References

- **ChemBERTa Paper**: "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction" (arXiv, 2020)
- **RoBERTa Paper**: "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (arXiv, 2019)
- **Organix13**: Large-scale organic molecules database
- **HuggingFace Transformers**: [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)

## ✅ Next Steps

1. **Start Training**: Run small model first to validate setup
2. **Monitor Progress**: Check Weights & Biases dashboard
3. **Fine-tune**: Adapt for specific molecular property prediction tasks
4. **Evaluate**: Test on molecular benchmarks (BBBP, Tox21, etc.)
5. **Scale Up**: Train medium/large models for better performance

## 🔬 Fine-tuning Example

```python
from transformers import RobertaForSequenceClassification, Trainer

# Load pre-trained model
model = RobertaForSequenceClassification.from_pretrained(
    "learning_source_20251210/compounds/chemberta2-output/chemberta2-small",
    num_labels=2  # Binary classification
)

# Fine-tune on your dataset
trainer = Trainer(
    model=model,
    train_dataset=your_train_dataset,
    eval_dataset=your_val_dataset,
    # ... other arguments
)
trainer.train()
```

---

**Implementation Date**: 2026-01-22
**Based On**: DNABERT-2, ESM-2, and RNAformer implementation patterns
**Dataset**: Organix13 SMILES compounds (~10M molecules)
**Architecture**: RoBERTa (improved BERT for chemical data)
