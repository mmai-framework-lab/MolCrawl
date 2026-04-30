#!/usr/bin/env bash
# Phase 5 - ChEBI-20 bidirectional generation evaluation.

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${CHEBI20_DIR:?CHEBI20_DIR must be set}"

ARCH="${ARCH:-gpt2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/chebi20}"

mkdir -p "$OUTPUT_DIR"

cmd=(python -m molcrawl.tasks.evaluation.chebi20
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality molecule_nat_lang
     --device "$DEVICE"
     --dataset-dir "$CHEBI20_DIR"
     --output-dir "$OUTPUT_DIR")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
"${cmd[@]}"
