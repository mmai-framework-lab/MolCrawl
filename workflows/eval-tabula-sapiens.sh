#!/usr/bin/env bash
# Phase 4 - Tabula Sapiens cell-type annotation.

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${TABULA_JSONL:?TABULA_JSONL must be set}"

ARCH="${ARCH:-rnaformer}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/tabula_sapiens}"

mkdir -p "$OUTPUT_DIR"

cmd=(python -m molcrawl.tasks.evaluation.tabula_sapiens
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality rna
     --device "$DEVICE"
     --jsonl-path "$TABULA_JSONL"
     --output-dir "$OUTPUT_DIR")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ -n "${HOLDOUT_TISSUES:-}" ]]; then
    cmd+=(--holdout-tissues $HOLDOUT_TISSUES)
fi
"${cmd[@]}"
