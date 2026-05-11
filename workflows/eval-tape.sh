#!/usr/bin/env bash
# Phase 2 - TAPE evaluation.

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${TAPE_DIR:?TAPE_DIR must be set}"

ARCH="${ARCH:-esm2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/tape}"
TASKS="${TASKS:-fluorescence stability remote_homology}"

mkdir -p "$OUTPUT_DIR"

for task in $TASKS; do
    cmd=(python -m molcrawl.tasks.evaluation.tape
         --model-path "$MODEL_PATH"
         --arch "$ARCH"
         --modality protein_sequence
         --device "$DEVICE"
         --task "$task"
         --task-dir "$TAPE_DIR/$task"
         --output-dir "$OUTPUT_DIR/$task")
    if [[ -n "${TOKENIZER_PATH:-}" ]]; then
        cmd+=(--tokenizer-path "$TOKENIZER_PATH")
    fi
    "${cmd[@]}"
done
