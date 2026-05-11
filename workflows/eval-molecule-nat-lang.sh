#!/usr/bin/env bash
# Phase 5 - molecule_nat_lang pair scoring.

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${PAIRS_CSV:?PAIRS_CSV must be set}"

ARCH="${ARCH:-gpt2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/molecule_nat_lang}"

mkdir -p "$OUTPUT_DIR"

cmd=(python -m molcrawl.tasks.evaluation.molecule_nat_lang
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality molecule_nat_lang
     --device "$DEVICE"
     --pairs-csv "$PAIRS_CSV"
     --output-dir "$OUTPUT_DIR")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
"${cmd[@]}"
