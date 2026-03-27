# Genome Sequence GPT-2 Compatibility Verification Report - `learning_20251104`

## Verification Time

2025-11-25 16:05

## Objective

Confirm that updates for the `molecule_nat_lang` dataset did not break existing `genome_sequence` GPT-2 training.

## Summary Result

### Successful Findings

1. **Core files remained unchanged**
   - `gpt2/train.py`
   - `gpt2/configs/genome_sequence/train_gpt2_config.py`
   - `gpt2/model.py`
   - `src/config/paths.py` (including `REFSEQ_DATASET_DIR`)

2. **Required data assets exist**
   - Raw data: `learning_20251104/genome_sequence/raw_files/` (~111GB)
   - Tokenizer: `learning_20251104/genome_sequence/spm_tokenizer.model`
   - HF cache: `learning_20251104/genome_sequence/hf_cache/` (~199GB)

3. **Backward compatibility preserved**
   - `PreparedDataset` updates are compatible with existing genome flow
   - Arrow-loading support for molecule_nat_lang does not interfere with genome training
   - `LEARNING_SOURCE_DIR` usage remains compatible

## Dataset State

- Total samples: 3,512,197
- Raw text size: ~118GB
- Chunk files: 235
- Tokenizer: SentencePiece

## Config Check (`train_gpt2_config.py`)

```python
tokenizer_path = get_refseq_tokenizer_path()
dataset_dir = REFSEQ_DATASET_DIR

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

max_iters = 600000
lr_decay_iters = 600000
warmup_iters = 200
learning_rate = 6e-6
min_lr = 6e-7

dataset = "genome_sequence"
dataset_params = {"dataset_dir": dataset_dir}
```

## Compatibility Analysis

### Code-path impact

- `PreparedDataset` was extended for Arrow and molecule-nl-specific handling
- Existing genome_sequence usage remains in the standard `PreparedDataset` flow
- No conflict detected with current genome training path

### Data-preparation status

`training_ready_hf_dataset` was reported as missing at verification time.
Raw files/cache/tokenizer are present, but preprocessing is required.

```bash
bash workflows/02-genome_sequence-prepare-gpt2.sh
```

or

```bash
LEARNING_SOURCE_DIR="learning_source" \
python src/genome_sequence/dataset/prepare_gpt2.py \
    assets/configs/genome_sequence.yaml
```

## Conclusion

No regression was detected in existing genome-sequence GPT-2 training logic due to molecule_nat_lang updates.
The remaining action is dataset preparation to recreate `training_ready_hf_dataset` before a full training run.
