# GPT-2 Training Verification Report - `learning_20251125`

## Verification Time

2025-11-25 15:31

## Objective

Verify that GPT-2 training runs correctly on the Molecule NL dataset in `learning_20251125`.

## Dataset Summary

```text
learning_20251125/molecule_nl/
├── arrow_splits/
│   ├── train.arrow/    (3,267,176 samples)
│   ├── test.arrow/     (30,344 samples)
│   └── valid.arrow/    (17,781 samples)
└── molecule_related_natural_language_tokenized.parquet (583MB)
```

- Total samples: 3,315,301
- Total tokens: ~342M
- Task types: 14

## Compatibility Checks

### BERT-style field check

Result: **PASS**

- `input_ids` present
- `attention_mask` present
- Types are valid

### GPT-2 batch/data loading check

Result: **PASS**

- Dataset load succeeded (`3,267,176` train samples)
- Tensor conversion succeeded
- Batch padding/truncation succeeded

## Training Test Configuration

```python
# Model
block_size = 128
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1

# Training
batch_size = 4
max_iters = 50
learning_rate = 3e-4
gradient_accumulation_steps = 1

# Data
dataset_dir = "learning_20251125/molecule_nl/arrow_splits"
vocab_size = 32008
```

## Observed Training Progress

| Iteration | Train Loss | Val Loss | Time (ms) | MFU   |
| --------- | ---------- | -------- | --------- | ----- |
| 0         | 10.0068    | 10.0237  | 2102.65   | -     |
| 10        | 7.3057     | 7.8500   | 147.49    | 0.59% |
| 20        | 6.4795     | 6.1294   | 252.51    | 0.91% |
| 30        | 5.8527     | 5.4260   | 215.43    | 0.74% |
| 40        | 5.0794     | 5.6638   | 55.20     | 0.68% |
| 50        | 4.6604     | 5.2967   | 188.08    | 0.73% |

## Result Analysis

- Train loss decreased from `10.01` to `4.66` (~53.5%)
- No runtime errors across 50 iterations
- Checkpoint saved successfully (~131MB)
- Validation path executed normally

## Generated Files

```text
test_gpt2_molecule_nl_20251125/
├── ckpt.pt       # model + optimizer states
└── logging.csv   # training log
```

## Conclusion

The `learning_20251125` Molecule NL dataset is compatible with GPT-2 training in this codebase.
