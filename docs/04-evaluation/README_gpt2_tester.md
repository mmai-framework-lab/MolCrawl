# GPT-2 Checkpoint Tester

This document explains how to test GPT-2 checkpoints, including loading, HF conversion, perplexity evaluation, and generation checks.

## Target Scripts

- `molcrawl/gpt2/test_checkpoint.py` (main)
- `molcrawl/gpt2/test_helper.py` (checkpoint discovery helper)
- `workflows/batch_test_gpt2.sh` (batch test runner)

## Single Checkpoint Test

```bash
python molcrawl/gpt2/test_checkpoint.py \
  --checkpoint_path "<LEARNING_SOURCE_DIR>/compounds/gpt2-output/compounds-small/ckpt.pt" \
  --domain compounds \
  --vocab_path assets/molecules/vocab.txt \
  --convert_to_hf \
  --output_dir gpt2_test_output
```

## Main Arguments

- `--checkpoint_path` (required): `.pt` checkpoint path
- `--output_dir`: output directory (default: `gpt2_test_output`)
- `--convert_to_hf`: convert to Hugging Face format
- `--test_dataset_params`: JSON string, for example `{"dataset_dir":"..."}`
- `--domain`: `compounds|molecule_nat_lang|genome|protein_sequence|rna`
- `--vocab_path`: vocab/model file path (used for some domains)
- `--max_test_samples`: max evaluation samples
- `--device`: for example `cuda`, `cpu`, `cuda:0`

## Examples

### Molecule NL

```bash
python molcrawl/gpt2/test_checkpoint.py \
  --checkpoint_path "<LEARNING_SOURCE_DIR>/molecule_nat_lang/gpt2-output/molecule_nat_lang-small/ckpt.pt" \
  --domain molecule_nat_lang \
  --test_dataset_params '{"dataset_dir":"<LEARNING_SOURCE_DIR>/molecule_nat_lang/training_ready_hf_dataset"}' \
  --max_test_samples 1000
```

### Protein Sequence

```bash
python molcrawl/gpt2/test_checkpoint.py \
  --checkpoint_path "<LEARNING_SOURCE_DIR>/protein_sequence/gpt2-output/protein_sequence-small/ckpt.pt" \
  --domain protein_sequence \
  --test_dataset_params '{"dataset_dir":"<LEARNING_SOURCE_DIR>/protein_sequence/training_ready_hf_dataset"}'
```

## Helper / Batch Execution

```bash
# list only
python molcrawl/gpt2/test_helper.py --search_dir . --list_only

# auto-run
python molcrawl/gpt2/test_helper.py --search_dir . --auto_run

# batch run
bash workflows/batch_test_gpt2.sh
bash workflows/batch_test_gpt2.sh /path/to/checkpoints
```

## Output

- `gpt2_test_report.json` (under `--output_dir`)
- `hf_model/` (when `--convert_to_hf` is set)

## Troubleshooting Checklist

- Check whether `--checkpoint_path` exists and is a valid `.pt` file.
- Check whether `--test_dataset_params` is valid JSON.
- For compounds, check whether `--vocab_path` is valid.
- Check whether `--device` matches your runtime environment.
