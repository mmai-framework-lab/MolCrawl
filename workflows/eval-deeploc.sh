#!/usr/bin/env bash
# Phase 2 - DeepLoc 2.0 subcellular localisation.
#
# Required:
#   MODEL_PATH       - encoder MLM checkpoint (esm2 / bert protein_sequence)
#   DEEPLOC_DATA     - path to the single-label CSV produced by
#                      `python -m molcrawl.tasks.evaluation.deeploc.prepare_csv`
#                      (the eval-data-deeploc.sh downloader builds it as
#                      $LEARNING_SOURCE_DIR/eval/deeploc/deeploc.csv)
#
# Optional:
#   ARCH                       - default esm2
#   TOKENIZER_PATH             - default uses arch+modality fallback
#   OUTPUT_DIR                 - default experiment_data/eval/deeploc
#   TEST_FRACTION              - default 0.2 (cluster split)
#   MAX_EXAMPLES               - class-balanced subsample size
#   SEED                       - default 0
#   BOOTSTRAP                  - default 100 (0 disables CI)
#   PREDICTIONS_PREVIEW_COUNT  - default 20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${DEEPLOC_DATA:?DEEPLOC_DATA must be set}"

ARCH="${ARCH:-esm2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/deeploc}"

mkdir -p "$OUTPUT_DIR"

cmd=("$PYTHON" -m molcrawl.tasks.evaluation.deeploc
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality protein_sequence
     --device "$DEVICE"
     --deeploc-data "$DEEPLOC_DATA"
     --output-dir "$OUTPUT_DIR")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ -n "${TEST_FRACTION:-}" ]]; then
    cmd+=(--test-fraction "$TEST_FRACTION")
fi
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
    cmd+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ -n "${SEED:-}" ]]; then
    cmd+=(--seed "$SEED")
fi
if [[ -n "${BOOTSTRAP:-}" ]]; then
    cmd+=(--bootstrap-samples "$BOOTSTRAP")
fi
if [[ -n "${PREDICTIONS_PREVIEW_COUNT:-}" ]]; then
    cmd+=(--predictions-preview-count "$PREDICTIONS_PREVIEW_COUNT")
fi
"${cmd[@]}"
