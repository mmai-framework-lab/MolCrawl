#!/usr/bin/env bash
# Phase 2 - ProteinGym zero-shot mutation effect evaluation.

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${PROTEINGYM_DATA:?PROTEINGYM_DATA must be set}"

ARCH="${ARCH:-gpt2}"
MODALITY="${MODALITY:-protein_sequence}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/proteingym}"

mkdir -p "$OUTPUT_DIR"

cmd=(python -m molcrawl.tasks.evaluation.proteingym
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality "$MODALITY"
     --device "$DEVICE"
     --proteingym-data "$PROTEINGYM_DATA"
     --output-dir "$OUTPUT_DIR")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
"${cmd[@]}"
