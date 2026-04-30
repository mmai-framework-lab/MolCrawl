# Molecule NL Training Guide

## Overview

This guide summarizes the current Molecule NL workflow and dataset compatibility for both BERT and GPT-2.

## Quick Flow

1. Prepare Molecule NL data.
2. Build the training-ready Hugging Face dataset.
3. Train BERT or GPT-2.
4. Run the compatibility check when needed.

## End-to-End Preparation

### Step 1: Prepare the Molecule NL dataset

```bash
export LEARNING_SOURCE_DIR="learning_source"
bash workflows/01-molecule_nat_lang-prepare.sh
```

This step runs `molcrawl/data/molecule_nat_lang/preparation.py` and creates:

- `${LEARNING_SOURCE_DIR}/molecule_nat_lang/molecule_related_natural_language_tokenized.parquet`
- `${LEARNING_SOURCE_DIR}/molecule_nat_lang/arrow_splits/` (split datasets)
- `${LEARNING_SOURCE_DIR}/molecule_nat_lang/gpt2_format/` (token-stream `.pt` files)

### Step 2: Build training-ready HF dataset

```bash
export LEARNING_SOURCE_DIR="learning_source"
bash workflows/02-molecule_nat_lang-prepare-gpt2.sh
```

This generates:

- `${LEARNING_SOURCE_DIR}/molecule_nat_lang/training_ready_hf_dataset`

Both current BERT and GPT-2 Molecule NL configs use this `training_ready_hf_dataset` as `dataset_dir`.

## Training

### BERT (Molecule NL)

Direct run:

```bash
python molcrawl/bert/main.py molcrawl/bert/configs/molecule_nat_lang.py
```

Workflow script:

```bash
bash workflows/03c-molecule_nat_lang-train-bert-small.sh
```

### GPT-2 (Molecule NL)

Direct run:

```bash
python molcrawl/gpt2/train.py molcrawl/gpt2/configs/molecule_nat_lang/train_gpt2_config.py
```

Workflow script:

```bash
bash workflows/03a-molecule_nat_lang-train-small.sh
```

## Data Compatibility

The Molecule NL preparation pipeline is designed so both models can train from the same source dataset.

Main tokenized fields include:

```python
{
  "input_ids": List[int],
  "attention_mask": List[int],
  "output_ids": List[int],
  "labels": List[int],
  "input_text": str,
  "real_input_text": str,
  "task_type": str,
  "valid_sample": bool,
  "input_too_long": bool,
}
```

## Compatibility Check

Before training, you can run:

```bash
export LEARNING_SOURCE_DIR="learning_source"
python molcrawl/preparation/test_molecule_nat_lang_compatibility.py
```

Expected summary includes:

- `BERT compatibility: PASS`
- `GPT-2 compatibility: PASS`

## Troubleshooting

### Dataset not found

- Confirm `LEARNING_SOURCE_DIR` is set.
- Re-run preparation:

```bash
bash workflows/01-molecule_nat_lang-prepare.sh
bash workflows/02-molecule_nat_lang-prepare-gpt2.sh
```

### Out-of-memory during training

- Reduce `batch_size` in config.
- Increase `gradient_accumulation_steps` to maintain effective batch size.
- For GPT-2, reduce `block_size` if needed.

### Low GPU utilization

- Increase `gradient_accumulation_steps` moderately.
- Use workflow scripts with built-in GPU selection/logging setup.

## Related Files

- BERT config: `molcrawl/bert/configs/molecule_nat_lang.py`
- GPT-2 config: `molcrawl/gpt2/configs/molecule_nat_lang/train_gpt2_config.py`
- Compatibility test: `molcrawl/preparation/test_molecule_nat_lang_compatibility.py`
- Dataset comparison report: `docs/07-reports/molecule_nat_lang_dataset_comparison_report.md`
