# LEARNING_SOURCE_DIR Environment Variable Migration Guide

## Overview

This update standardizes dataset download/preprocessing outputs under `LEARNING_SOURCE_DIR` to prevent file generation directly under the Git repository root.

## Change Date

November 4, 2025

## What Changed

### 1. Mandatory environment variable

`LEARNING_SOURCE_DIR` is now required.

If it is not set, scripts show an error like:

```text
ERROR: LEARNING_SOURCE_DIR environment variable is not set.
Please set it before running this script:
  export LEARNING_SOURCE_DIR=/path/to/learning_source
  # or
  LEARNING_SOURCE_DIR=learning_20251104 python <script>
```

### 2. Recommended directory layout

```text
$LEARNING_SOURCE_DIR/
├── compounds/
│   ├── data/       # Dataset storage
│   ├── logs/       # Log files
│   └── report/     # Evaluation reports
├── genome_sequence/
│   ├── data/
│   │   ├── clinvar/
│   │   ├── cosmic/
│   │   └── omim/
│   ├── logs/
│   └── report/
├── protein_sequence/
│   ├── data/
│   │   └── proteingym/
│   ├── logs/
│   └── report/
├── rna/
│   ├── data/
│   ├── logs/
│   └── report/
└── molecule_nl/
    ├── data/
    ├── logs/
    └── report/
```

### 3. Updated files

#### Data preparation scripts

| File                                                     | Change                                  | New Default Output Path                               |
| -------------------------------------------------------- | --------------------------------------- | ----------------------------------------------------- |
| `scripts/evaluation/bert/proteingym_data_preparation.py` | `LEARNING_SOURCE_DIR` required          | `$LEARNING_SOURCE_DIR/protein_sequence/data/proteingym` |
| `scripts/evaluation/gpt2/omim_data_preparation.py`       | `LEARNING_SOURCE_DIR` required          | `$LEARNING_SOURCE_DIR/genome_sequence/data/omim`      |
| `scripts/evaluation/gpt2/cosmic_data_preparation.py`     | `LEARNING_SOURCE_DIR` required          | `$LEARNING_SOURCE_DIR/genome_sequence/data/cosmic`    |

#### Utility

| File                              | Change                                                                 |
| --------------------------------- | ---------------------------------------------------------------------- |
| `src/utils/evaluation_output.py`  | `get_learning_source_dir()` now exits with error if env var is unset |

### 4. Existing evaluation scripts

Scripts already using `utils.evaluation_output` automatically inherit this behavior:

- `scripts/evaluation/bert/clinvar_evaluation.py`
- `scripts/evaluation/gpt2/clinvar_evaluation.py`
- `scripts/evaluation/bert/molecule_nl_evaluation.py`
- `scripts/evaluation/gpt2/molecule_nl_evaluation.py`
- `scripts/evaluation/bert/proteingym_evaluation.py`
- `scripts/evaluation/gpt2/proteingym_evaluation.py`

## Usage

### Set the environment variable

#### Option 1: export (recommended)

```bash
export LEARNING_SOURCE_DIR=/path/to/learning_source
# or
export LEARNING_SOURCE_DIR=learning_20251104
```

#### Option 2: inline assignment

```bash
LEARNING_SOURCE_DIR=learning_20251104 python scripts/evaluation/bert/proteingym_data_preparation.py --sample_only
```

### Tested example

```bash
# 1) Create a test directory structure
mkdir -p learning_20251104/{protein_sequence,genome_sequence,compounds,rna,molecule_nl}/{data,logs,report}

# 2) Create ProteinGym sample data
LEARNING_SOURCE_DIR=learning_20251104 python scripts/evaluation/bert/proteingym_data_preparation.py --sample_only

# 3) Create OMIM sample data
LEARNING_SOURCE_DIR=learning_20251104 python scripts/evaluation/gpt2/omim_data_preparation.py --mode sample --num_samples 50

# 4) Create COSMIC sample data
LEARNING_SOURCE_DIR=learning_20251104 python scripts/evaluation/gpt2/cosmic_data_preparation.py --create_sample_data --max_samples 30
```

## Migration Checklist

- [x] Require `LEARNING_SOURCE_DIR` in data preparation scripts
- [x] Reorganize default outputs by model type
- [x] Move logs under `LEARNING_SOURCE_DIR`
- [x] Improve error messages
- [x] Validate with test runs

## Notes

1. **Compatibility**: Existing evaluation scripts inherit new behavior through `utils.evaluation_output`.
2. **Log location**: Logs are saved under `$LEARNING_SOURCE_DIR/<model_type>/logs/`.
3. **Report output**: Reports are saved under `$LEARNING_SOURCE_DIR/<model_type>/report/` with timestamps.
4. **Backward compatibility**: You can still write to custom paths with `--output_dir`.
