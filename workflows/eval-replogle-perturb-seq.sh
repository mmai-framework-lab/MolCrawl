#!/usr/bin/env bash
# Phase 4 - Replogle Perturb-seq evaluation.

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${REPLOGLE_DATA:?REPLOGLE_DATA must be set}"

ARCH="${ARCH:-rnaformer}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/replogle_perturb_seq}"

mkdir -p "$OUTPUT_DIR"

cmd=(python -m molcrawl.tasks.evaluation.replogle_perturb_seq
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality rna
     --device "$DEVICE"
     --replogle-data "$REPLOGLE_DATA"
     --output-dir "$OUTPUT_DIR")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
"${cmd[@]}"
