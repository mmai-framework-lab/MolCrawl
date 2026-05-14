#!/usr/bin/env bash
# Phase 2 - TAPE evaluation.
#
# Required:
#   MODEL_PATH       - encoder MLM checkpoint (esm2 / bert protein_sequence)
#   TAPE_DIR         - directory containing per-task subdirs of JSONL splits
#                      (built by workflows/data/eval-data-tape.sh)
#
# Optional:
#   ARCH                       - default esm2
#   TOKENIZER_PATH             - default uses arch+modality fallback
#   OUTPUT_DIR                 - default experiment_data/eval/tape
#   TASKS                      - default "fluorescence stability remote_homology"
#   MAX_EXAMPLES               - task-aware stratified subsample size per split
#   SEED                       - default 42
#   BOOTSTRAP                  - default 100 (0 disables CI)
#   PREDICTIONS_PREVIEW_COUNT  - default 20
#   CONTACT_MIN_SEPARATION     - long-range threshold for contact_prediction (default 24)
#   CONTACT_PAIRS_PER_PROTEIN  - training pairs per protein for contact_prediction (default 50)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${TAPE_DIR:?TAPE_DIR must be set}"

ARCH="${ARCH:-esm2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/tape}"
TASKS="${TASKS:-fluorescence stability remote_homology}"

mkdir -p "$OUTPUT_DIR"

for task in $TASKS; do
    cmd=("$PYTHON" -m molcrawl.tasks.evaluation.tape
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
    if [[ -n "${CONTACT_MIN_SEPARATION:-}" ]]; then
        cmd+=(--contact-min-separation "$CONTACT_MIN_SEPARATION")
    fi
    if [[ -n "${CONTACT_PAIRS_PER_PROTEIN:-}" ]]; then
        cmd+=(--contact-pairs-per-protein "$CONTACT_PAIRS_PER_PROTEIN")
    fi
    "${cmd[@]}"
done
