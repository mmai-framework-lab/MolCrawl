#!/usr/bin/env bash
# Phase 1 - ChEMBL scaffold held-out evaluation.
#
# Required environment:
#   MODEL_PATH      - decoder or encoder checkpoint
#   TOKENIZER_PATH  - SentencePiece tokenizer
#   HELDOUT_CSV     - path to the scaffold held-out CSV (built by
#                     `python -m molcrawl.tasks.evaluation.chembl_scaffold_heldout.prepare_csv`)
#
# Optional:
#   ARCH                       - default gpt2
#   LABEL_COLUMN               - set for encoder probe mode
#   TRAIN_CSV                  - required for encoder probe mode
#   OUTPUT_DIR                 - default experiment_data/eval/chembl_scaffold_heldout
#   SMILES_COLUMN              - default 'smiles'
#   MAX_EXAMPLES               - length-stratified subsample size (default unset = full split)
#   SEED                       - reproducibility seed (default 42)
#   BOOTSTRAP                  - bootstrap resamples (default 100; 0 disables)
#   PREDICTIONS_PREVIEW_COUNT  - narrative preview size (default 30)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${TOKENIZER_PATH:?TOKENIZER_PATH must be set}"
: "${HELDOUT_CSV:?HELDOUT_CSV must be set}"

ARCH="${ARCH:-gpt2}"
DEVICE="${DEVICE:-cuda}"
RUNTAG="${RUNTAG:-chembl_scaffold_heldout_default}"
OUTPUT_DIR="${OUTPUT_DIR:-$(compose_eval_output_dir compounds "$MODEL_PATH" "$RUNTAG")}"
SMILES_COLUMN="${SMILES_COLUMN:-smiles}"

mkdir -p "$OUTPUT_DIR"

cmd=("$PYTHON" -m molcrawl.tasks.evaluation.chembl_scaffold_heldout
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
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
    cmd+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ -n "${SEED:-}" ]]; then
    cmd+=(--seed "$SEED")
fi
if [[ -n "${BOOTSTRAP:-}" ]]; then
    cmd+=(--bootstrap-samples "$BOOTSTRAP")
fi
if [[ -n "${PREDICTIONS_PREVIEW_COUNT:-}" ]]; then
    cmd+=(--predictions-preview-count "$PREDICTIONS_PREVIEW_COUNT")
fi

"${cmd[@]}"
