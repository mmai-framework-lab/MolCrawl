# BERT Checkpoint Tester

This document explains how to use the BERT checkpoint tester for loading, inference, MLM checks, and basic evaluation.

## Target Scripts

- `molcrawl/models/bert/test_checkpoint.py`

## Basic Run

```bash
python molcrawl/models/bert/test_checkpoint.py \
  --checkpoint_path "<LEARNING_SOURCE_DIR>/compounds/bert-output/compounds-small/checkpoint-1000" \
  --domain compounds \
  --vocab_path assets/molecules/vocab.txt
```

`--checkpoint_path` should point to a checkpoint directory loadable via `from_pretrained`.

## Main Arguments

- `--checkpoint_path` (required): target checkpoint
- `--domain`: `compounds|molecule_nat_lang|genome|protein_sequence|rna`
- `--vocab_path`: required for some domains such as compounds/genome
- `--dataset_path`: optional; runs dataset-based evaluation if provided
- `--test_texts`: optional; overrides default test samples

## Examples

### Molecule NL

```bash
python molcrawl/models/bert/test_checkpoint.py \
  --checkpoint_path "<LEARNING_SOURCE_DIR>/molecule_nat_lang/bert-output/molecule_nat_lang-small/checkpoint-1000" \
  --domain molecule_nat_lang
```

### Genome

```bash
python molcrawl/models/bert/test_checkpoint.py \
  --checkpoint_path "<LEARNING_SOURCE_DIR>/genome_sequence/bert-output/genome_sequence-small/checkpoint-1000" \
  --domain genome \
  --vocab_path "<LEARNING_SOURCE_DIR>/genome_sequence/spm_tokenizer.model"
```

## Output

- `test_report.json` is saved under the parent directory of `--checkpoint_path`.

## Troubleshooting Checklist

- Check whether `--checkpoint_path` is valid.
- Check whether `--domain` is one of the supported choices.
- For compounds/genome, check whether `--vocab_path` points to a valid file.
- Check required dependencies (`torch`, `transformers`, `datasets`).

## Reference

- Sample vocab generator: `workflows/create_sample_vocab.sh`
