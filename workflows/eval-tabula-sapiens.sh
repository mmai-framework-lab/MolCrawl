#!/usr/bin/env bash
# Phase 4 - Tabula Sapiens cell-type annotation.
#
# Required:
#   MODEL_PATH       - encoder MLM checkpoint (bert rna)
#   TABULA_JSONL     - path to the tokenised-cell JSONL produced by
#                      workflows/data/eval-data-tabula-sapiens.sh
#
# Optional:
#   ARCH                       - default gpt2
#   TOKENIZER_PATH             - default uses arch+modality fallback
#   OUTPUT_DIR                 - default experiment_data/eval/tabula_sapiens
#   TEST_FRACTION              - default 0.2 (random split)
#   HOLDOUT_TISSUES            - space-separated tissue names; switches to
#                                cross-tissue split (overrides TEST_FRACTION)
#   MAX_CELLS                  - class-balanced subsample cap
#   SEED                       - default 0
#   BOOTSTRAP                  - default 100 (0 disables CI)
#   PREDICTIONS_PREVIEW_COUNT  - default 16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${TABULA_JSONL:?TABULA_JSONL must be set}"

ARCH="${ARCH:-gpt2}"
DEVICE="${DEVICE:-cuda}"
RUNTAG="${RUNTAG:-tabula_sapiens_default}"
OUTPUT_DIR="${OUTPUT_DIR:-$(compose_eval_output_dir rna "$MODEL_PATH" "$RUNTAG")}"

mkdir -p "$OUTPUT_DIR"

cmd=("$PYTHON" -m molcrawl.tasks.evaluation.tabula_sapiens
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
if [[ -n "${TEST_FRACTION:-}" ]]; then
    cmd+=(--test-fraction "$TEST_FRACTION")
fi
if [[ -n "${MAX_CELLS:-}" ]]; then
    cmd+=(--max-cells "$MAX_CELLS")
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
