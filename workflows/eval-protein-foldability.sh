#!/usr/bin/env bash
# Phase 2 - structure-free foldability proxies.

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${REFERENCE_FASTA:?REFERENCE_FASTA must be set}"

ARCH="${ARCH:-gpt2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/protein_foldability}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"

mkdir -p "$OUTPUT_DIR"

cmd=(python -m molcrawl.tasks.evaluation.protein_foldability
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality protein_sequence
     --device "$DEVICE"
     --reference-fasta "$REFERENCE_FASTA"
     --output-dir "$OUTPUT_DIR"
     --num-samples "$NUM_SAMPLES")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
"${cmd[@]}"
