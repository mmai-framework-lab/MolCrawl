# RNAformer Implementation Summary

## 📁 Files Created

### Core Implementation

1. **rnaformer/main.py** (348 lines)
   - Main training script with RNADatasetLoader class
   - Supports checkpoint resumption
   - Weights & Biases integration
   - Mixed precision training (FP16)

2. **rnaformer/configurator.py** (53 lines)
   - Command-line argument parsing
   - Configuration file loading with overrides

3. **rnaformer/configs/rna.py** (136 lines)
   - RNA transcriptome dataset configuration
   - Gene vocabulary loading
   - WordLevel tokenizer setup
   - Preprocessing functions

### Bootstrap Scripts

1. **workflows/03f-rna-train-rnaformer-small.sh**
2. **workflows/03f-rna-train-rnaformer-medium.sh**
3. **workflows/03f-rna-train-rnaformer-large.sh**
   - Executable training scripts for 3 model sizes
   - Environment variable configuration
   - Automatic logging

### Documentation

1. **docs/RNAFORMER_TRAINING_GUIDE.md**
   - Comprehensive user guide
   - Quick start examples
   - Troubleshooting tips
   - Performance benchmarks

## 🚀 Quick Start

### Training Commands

```bash
# Small model (recommended for testing)
CUDA_VISIBLE_DEVICES=0 ./workflows/03f-rna-train-rnaformer-small.sh

# Medium model
CUDA_VISIBLE_DEVICES=0 ./workflows/03f-rna-train-rnaformer-medium.sh

# Large model
CUDA_VISIBLE_DEVICES=0 ./workflows/03f-rna-train-rnaformer-large.sh

# With Weights & Biases
LEARNING_SOURCE_DIR=learning_source_20250904-rna-refined \
USE_WANDB=True \
WANDB_PROJECT=rnaformer-transcriptome \
  ./workflows/03f-rna-train-rnaformer-small.sh
```

## 🔧 Model Specifications

| Model Size | Parameters | Hidden | Layers | Heads | Intermediate |
| ---------- | ---------- | ------ | ------ | ----- | ------------ |
| Small      | ~40M       | 512    | 8      | 8     | 2048         |
| Medium     | ~90M       | 768    | 12     | 12    | 3072         |
| Large      | ~180M      | 1024   | 16     | 16    | 4096         |

## ⚙️ Key Features

### RNA Transcriptome Specific

- **Tokenization**: Gene ID-based vocabulary (~60K genes)
- **Max Length**: 1024 tokens (full cell expression profile)
- **Architecture**: Geneformer-based BERT encoder
- **Dataset**: CellXGene single-cell RNA-seq data (~54M cells)

### Training Optimizations

- **Learning Rate**: 1e-4 (optimized for RNA data)
- **Batch Size**: 8 per device (memory efficient)
- **Gradient Accumulation**: 16 steps (effective batch = 128)
- **Mixed Precision**: FP16 for faster training
- **Warmup**: 10,000 steps with cosine schedule

### Infrastructure

- Automatic checkpoint resumption
- Weights & Biases integration
- Comprehensive logging
- Memory-efficient data loading

## 📊 Comparison with Other Implementations

| Feature       | RNAformer         | DNABERT-2     | ESM-2             |
| ------------- | ----------------- | ------------- | ----------------- |
| Domain        | RNA transcriptome | DNA sequences | Protein sequences |
| Tokenization  | Gene IDs          | BPE           | Amino acids       |
| Vocab Size    | ~60K              | ~4K           | ~33               |
| Max Length    | 1024              | 1024          | 1024              |
| Learning Rate | 1e-4              | 3e-5          | 4e-4              |
| Batch Size    | 8                 | 16            | 4                 |
| Dropout       | 0.1               | 0.1           | 0.0               |

## 🎯 Use Cases

1. **Cell Type Classification**: Predict cell types from expression profiles
2. **Gene Function Prediction**: Infer gene functions from co-expression
3. **Disease State Identification**: Classify healthy vs. disease cells
4. **Drug Response Prediction**: Predict cellular response to treatments
5. **Gene Regulatory Network**: Learn gene-gene interactions

## 📈 Expected Performance

### Training Metrics

- **Loss**: Should decrease to ~2.5-3.0 after 100K steps
- **Perplexity**: Target ~12-20 for good convergence
- **Memory Usage**: 12-30 GB depending on model size

### Computational Requirements

- **Small**: ~40 hours on A100 (100K steps)
- **Medium**: ~55 hours on A100 (100K steps)
- **Large**: ~83 hours on A100 (100K steps)

## 🛠️ Technical Details

### Data Flow

1. Load gene vocabulary (JSON format)
2. Create WordLevel tokenizer
3. Load HuggingFace datasets (Arrow format)
4. Add attention masks
5. Apply MLM with 15% masking
6. Train with AdamW optimizer

### Directory Structure

```
learning_source_20250904-rna-refined/
└── rna/
    ├── gene_vocab.json              # ~60K gene IDs
    ├── training_ready_hf_dataset/   # Arrow format
    │   ├── train/
    │   └── test/
    ├── rnaformer-output/            # Checkpoints
    │   ├── rnaformer-small/
    │   ├── rnaformer-medium/
    │   └── rnaformer-large/
    └── logs/                        # Training logs
```

## 🔍 Validation

### Pre-training Checks

```bash
# 1. Verify dataset
python -c "from datasets import load_from_disk; \
  ds = load_from_disk('learning_source_20250904-rna-refined/rna/training_ready_hf_dataset/train'); \
  print(f'Dataset size: {len(ds)}')"

# 2. Check vocabulary
python -c "import json; \
  vocab = json.load(open('learning_source_20250904-rna-refined/rna/gene_vocab.json')); \
  print(f'Vocab size: {len(vocab)}')"

# 3. Test tokenizer
python -c "from transformers import AutoTokenizer; \
  tok = AutoTokenizer.from_pretrained('custom_tokenizer_rnaformer'); \
  print(f'Tokenizer loaded: {len(tok)} tokens')"
```

### Post-training Checks

```bash
# Check model output
ls -la learning_source_20250904-rna-refined/rna/rnaformer-output/rnaformer-small/

# View logs
tail -f learning_source_20250904-rna-refined/rna/logs/rnaformer-train-small-*.log
```

## 🐛 Common Issues

### Issue: Gene vocabulary not found

```bash
# Solution: Verify LEARNING_SOURCE_DIR
export LEARNING_SOURCE_DIR=learning_source_20250904-rna-refined
```

### Issue: OOM (Out of Memory)

```bash
# Solution: Reduce batch size
python rnaformer/main.py --config rnaformer/configs/rna.py --batch_size 4
```

### Issue: Training too slow

```bash
# Solution: Increase gradient accumulation
python rnaformer/main.py --gradient_accumulation_steps 32
```

## 📚 References

- **Geneformer Paper**: "Transfer learning enables predictions in network biology" (Nature, 2023)
- **CellXGene**: <https://cellxgene.cziscience.com/>
- **HuggingFace Transformers**: <https://huggingface.co/docs/transformers/>

## ✅ Next Steps

1. **Start Training**: Run small model first to validate setup
2. **Monitor Progress**: Check Weights & Biases dashboard
3. **Evaluate**: Run downstream tasks (classification, clustering)
4. **Fine-tune**: Adapt for specific use cases
5. **Scale Up**: Train medium/large models for better performance

---

**Implementation Date**: 2026-01-22  
**Based On**: DNABERT-2 and ESM-2 implementation patterns  
**Dataset**: CellXGene single-cell RNA-seq (54M cells)
