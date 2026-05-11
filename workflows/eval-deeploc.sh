#!/usr/bin/env bash
# Phase 2 - DeepLoc 2.0 subcellular localisation.

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${DEEPLOC_DATA:?DEEPLOC_DATA must be set}"

ARCH="${ARCH:-esm2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/deeploc}"

mkdir -p "$OUTPUT_DIR"

cmd=(python -m molcrawl.tasks.evaluation.deeploc
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality protein_sequence
     --device "$DEVICE"
     --deeploc-data "$DEEPLOC_DATA"
     --output-dir "$OUTPUT_DIR")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
"${cmd[@]}"
