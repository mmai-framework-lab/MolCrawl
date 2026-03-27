# BERT Training Verification Report

## Verification Time

2025-11-25 15:52

## Objective

Verify that BERT training runs correctly on the Molecule NL dataset.

## Dataset Summary

```text
learning_20251125/molecule_nat_lang/
├── arrow_splits/
│   ├── train.arrow/    (3,267,176 samples)
│   ├── test.arrow/     (30,344 samples)
│   └── valid.arrow/    (17,781 samples)
└── molecule_related_natural_language_tokenized.parquet (583MB)
```

- Total samples: 3,315,301
- Total tokens: ~342M
- Task types: 14
- Training split used: 3,267,176
- Eval subset used: 10,000 (limited for speed)

## Test Configuration

```python
# Model
model_size = "small"        # ~109M params
max_length = 128
vocab_size = 32008

# Training
batch_size = 4
gradient_accumulation_steps = 1
per_device_eval_batch_size = 4
max_steps = 50
learning_rate = 6e-5
weight_decay = 0.1
warmup_steps = 10
log_interval = 10
mlm_probability = 0.15
```

MLM strategy:

- 80% replace with `[MASK]`
- 10% replace with random token
- 10% keep original token

## Training/Evaluation Progress

### Training

| Step | Train Loss | Learning Rate | Gradient Norm |
| ---- | ---------- | ------------- | ------------- |
| 10   | 9.2835     | 6.00e-05      | 10.24         |
| 20   | 7.7051     | 4.50e-05      | -             |
| 30   | 7.2210     | 3.00e-05      | -             |
| 40   | 6.6476     | 1.50e-05      | -             |
| 50   | 6.5095     | 0.00e+00      | 6.55          |

### Evaluation

| Step | Eval Loss | Eval Runtime | Samples/sec | Epoch |
| ---- | --------- | ------------ | ----------- | ----- |
| 10   | 8.2294    | 74.23s       | 134.71      | 0.0   |
| 20   | 7.5155    | 74.60s       | 134.06      | 0.0   |
| 30   | 7.0527    | 64.47s       | 155.11      | 0.0   |
| 40   | 6.7736    | 68.34s       | 146.33      | 0.0   |
| 50   | 6.6860    | 71.28s       | 140.29      | 0.0   |

## Result Analysis

- Train loss decreased: `9.28 -> 6.51` (~29.8%)
- Eval loss decreased: `8.23 -> 6.69` (~18.7%)
- 50 steps completed without runtime errors
- Gradient norm remained stable
- Throughput remained stable around ~140 samples/sec during eval

## Checklist

- [x] Arrow loading works
- [x] Model initialization works
- [x] MLM data collator works
- [x] Padding/truncation works
- [x] Forward/backward pass works
- [x] Loss/optimizer/scheduler work
- [x] Checkpoint save works
- [x] Periodic evaluation works

## Conclusion

BERT training on `learning_20251125` is operational and stable for Molecule NL data under the tested setup.
