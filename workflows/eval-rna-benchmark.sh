#!/usr/bin/env bash
# Phase 4 - rna_benchmark evaluation (migration).

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${RNA_JSONL:?RNA_JSONL must be set}"

ARCH="${ARCH:-bert}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/rna_benchmark}"

mkdir -p "$OUTPUT_DIR"

cmd=(python -m molcrawl.tasks.evaluation.rna_benchmark
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality rna
     --device "$DEVICE"
     --rna-jsonl "$RNA_JSONL"
     --output-dir "$OUTPUT_DIR")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
"${cmd[@]}"
