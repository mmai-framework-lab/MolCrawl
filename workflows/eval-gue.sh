#!/usr/bin/env bash
# Phase 3 - GUE 28-task genome classification evaluation.

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${GUE_DIR:?GUE_DIR must be set}"

ARCH="${ARCH:-dnabert2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/gue}"
TASKS="${TASKS:-}"

mkdir -p "$OUTPUT_DIR"

if [[ -z "$TASKS" ]]; then
    TASKS=$(python -c "from molcrawl.tasks.evaluation.gue.data_preparation import TASKS; print(' '.join(TASKS))")
fi

for task in $TASKS; do
    task_dir="$GUE_DIR/$task"
    if [[ ! -d "$task_dir" ]]; then
        echo "Skipping missing $task_dir" >&2
        continue
    fi
    cmd=(python -m molcrawl.tasks.evaluation.gue
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
    "${cmd[@]}"
done
