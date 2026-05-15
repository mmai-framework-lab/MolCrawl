#!/usr/bin/env bash
# Phase 3 - GUE 28-task genome classification evaluation.
#
# Required:
#   MODEL_PATH       - encoder MLM checkpoint (dnabert2 / bert genome_sequence)
#   GUE_DIR          - directory containing 28 sub-task folders
#                      (built by workflows/data/eval-data-gue.sh)
#
# Optional:
#   ARCH                       - default dnabert2
#   TOKENIZER_PATH             - default uses arch+modality fallback
#   OUTPUT_DIR                 - default experiment_data/eval/gue
#   TASKS                      - space-separated subset (default: all 28)
#   MAX_EXAMPLES               - class-balanced subsample size per split
#   SEED                       - default 42
#   BOOTSTRAP                  - default 100 (0 disables CI)
#   PREDICTIONS_PREVIEW_COUNT  - default 16
#
# To run a quick smoke set on a couple of small tasks:
#   TASKS="prom_300_all H3" MAX_EXAMPLES=300 BOOTSTRAP=30 bash workflows/eval-gue.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${GUE_DIR:?GUE_DIR must be set}"

ARCH="${ARCH:-dnabert2}"
DEVICE="${DEVICE:-cuda}"
RUNTAG="${RUNTAG:-gue_default}"
OUTPUT_DIR="${OUTPUT_DIR:-$(compose_eval_output_dir genome_sequence "$MODEL_PATH" "$RUNTAG")}"
TASKS="${TASKS:-}"

mkdir -p "$OUTPUT_DIR"

if [[ -z "$TASKS" ]]; then
    TASKS=$("$PYTHON" -c "from molcrawl.tasks.evaluation.gue.data_preparation import TASKS; print(' '.join(TASKS))")
fi

for task in $TASKS; do
    task_dir="$GUE_DIR/$task"
    if [[ ! -d "$task_dir" ]]; then
        echo "Skipping missing $task_dir" >&2
        continue
    fi
    cmd=("$PYTHON" -m molcrawl.tasks.evaluation.gue
         --model-path "$MODEL_PATH"
         --arch "$ARCH"
         --modality genome_sequence
         --device "$DEVICE"
         --task "$task"
         --task-dir "$task_dir"
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
    "${cmd[@]}"
done
