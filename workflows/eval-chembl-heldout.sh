#!/usr/bin/env bash
# Phase 1 - ChEMBL scaffold held-out evaluation.
#
# Required environment:
#   MODEL_PATH      - decoder or encoder checkpoint
#   TOKENIZER_PATH  - SentencePiece tokenizer
#   HELDOUT_CSV     - path to the scaffold held-out CSV
#
# Optional:
#   ARCH            - default gpt2
#   LABEL_COLUMN    - set for encoder probe mode
#   TRAIN_CSV       - required for encoder probe mode
#   OUTPUT_DIR      - default experiment_data/eval/chembl_scaffold_heldout

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${TOKENIZER_PATH:?TOKENIZER_PATH must be set}"
: "${HELDOUT_CSV:?HELDOUT_CSV must be set}"

ARCH="${ARCH:-gpt2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/chembl_scaffold_heldout}"
SMILES_COLUMN="${SMILES_COLUMN:-smiles}"

mkdir -p "$OUTPUT_DIR"

cmd=(python -m molcrawl.tasks.evaluation.chembl_scaffold_heldout
     --model-path "$MODEL_PATH"
     --tokenizer-path "$TOKENIZER_PATH"
     --arch "$ARCH"
     --modality compounds
     --device "$DEVICE"
     --heldout-csv "$HELDOUT_CSV"
     --smiles-column "$SMILES_COLUMN"
     --output-dir "$OUTPUT_DIR")

if [[ -n "${LABEL_COLUMN:-}" ]]; then
    : "${TRAIN_CSV:?TRAIN_CSV must be set when LABEL_COLUMN is provided}"
    cmd+=(--label-column "$LABEL_COLUMN" --train-csv "$TRAIN_CSV")
fi

"${cmd[@]}"
