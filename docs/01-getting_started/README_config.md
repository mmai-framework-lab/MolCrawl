# Path Configuration Management

## Overview

This document explains the configuration files used to centrally manage dataset paths across the project.

## Configuration Files

### Python Configuration

- `molcrawl/config/paths.py`: path constants for Python scripts

### Shell Configuration

- `molcrawl/config/env.sh`: environment variable settings for shell scripts

## Usage

### Usage in Python Scripts

```python
#!/usr/bin/env python3
from molcrawl.config.paths import UNIPROT_DATASET_DIR, REFSEQ_DATASET_DIR

# Load a dataset
from datasets import load_from_disk
dataset = load_from_disk(UNIPROT_DATASET_DIR)
```

### Usage in Shell Scripts

```bash
#!/bin/bash

# Load configuration
source molcrawl/config/env.sh

# Build paths from LEARNING_SOURCE_DIR
UNIPROT_DATASET_DIR="$LEARNING_SOURCE_DIR/protein_sequence/training_ready_hf_dataset"
REFSEQ_DATASET_DIR="$LEARNING_SOURCE_DIR/genome_sequence/training_ready_hf_dataset"

echo "Learning source: $LEARNING_SOURCE_DIR"
echo "UniProt dataset dir: $UNIPROT_DATASET_DIR"
echo "RefSeq dataset dir: $REFSEQ_DATASET_DIR"
```

## How to Change Paths

To change the dataset storage directory:

1. Update `LEARNING_SOURCE_DIR` in `molcrawl/config/env.sh`.
2. If needed, override it in your current shell with `export LEARNING_SOURCE_DIR=...`.

Example:

```bash
# molcrawl/config/env.sh
export LEARNING_SOURCE_DIR="learning_source"
```

## Available Constants and Environment Variables

### Python (paths.py)

- `LEARNING_SOURCE_DIR`: base directory name
- `PROTEIN_SEQUENCE_DIR`: Protein Sequence directory path
- `GENOME_SEQUENCE_DIR`: Genome Sequence directory path
- `RNA_DATASET_DIR`: RNA directory path
- `MOLECULE_NAT_LANG_DIR`: Molecule_Nat_Lang directory path
- `COMPOUNDS_DIR`: Compounds directory path
- `UNIPROT_DATASET_DIR`: UniProt dataset path
- `REFSEQ_DATASET_DIR`: RefSeq dataset path
- `CELLXGENE_DATASET_DIR`: CellxGene dataset path
- `COMPOUNDS_DATASET_DIR`: Compounds dataset path (includes `organix13_tokenized.parquet`)
- `MOLECULE_NAT_LANG_DATASET_DIR`: Molecule_Nat_Lang dataset path (includes `molecule_related_natural_language_tokenized.parquet`)
- `ABSOLUTE_LEARNING_SOURCE_PATH`: absolute path of the base directory

### Shell (env.sh)

- `$LEARNING_SOURCE_DIR`: base directory name

## Mapping from Legacy Shell Variables

`env.sh` exports only `$LEARNING_SOURCE_DIR`.
Legacy variables should be derived from `$LEARNING_SOURCE_DIR` as follows:

- `$UNIPROT_DATASET_DIR` → `$LEARNING_SOURCE_DIR/protein_sequence/training_ready_hf_dataset`
- `$REFSEQ_DATASET_DIR` → `$LEARNING_SOURCE_DIR/genome_sequence/training_ready_hf_dataset`
- `$CELLXGENE_DATASET_DIR` → `$LEARNING_SOURCE_DIR/rna/training_ready_hf_dataset`
- `$COMPOUNDS_DATASET_DIR` → `$LEARNING_SOURCE_DIR/compounds/organix13/compounds/training_ready_hf_dataset`
- `$MOLECULE_NAT_LANG_DATASET_DIR` → `$LEARNING_SOURCE_DIR/molecule_nat_lang/training_ready_hf_dataset`
