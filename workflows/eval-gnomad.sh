#!/usr/bin/env bash
# Phase 3 - gnomAD allele-frequency correlation.

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${GNOMAD_DATA:?GNOMAD_DATA must be set}"

ARCH="${ARCH:-gpt2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/gnomad_af_correlation}"

mkdir -p "$OUTPUT_DIR"

cmd=(python -m molcrawl.tasks.evaluation.gnomad_af_correlation
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality genome_sequence
     --device "$DEVICE"
     --gnomad-data "$GNOMAD_DATA"
     --output-dir "$OUTPUT_DIR")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
"${cmd[@]}"
