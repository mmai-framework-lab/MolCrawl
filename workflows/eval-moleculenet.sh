#!/usr/bin/env bash
# Phase 1 - MoleculeNet property-prediction evaluation.
#
# Required environment:
#   MODEL_PATH      - embedding-capable checkpoint (ChemBERTa-2, BERT compound, ...)
#   TOKENIZER_PATH  - tokenizer file when applicable (empty for HF built-ins)
#   MOLECULENET_DIR - LEARNING_SOURCE_DIR/eval/moleculenet or a subset root
#
# Optional:
#   ARCH            - architecture tag (default: chemberta2)
#   MODALITY        - foundation modality (default: compounds)
#   OUTPUT_DIR      - default ${LEARNING_SOURCE_DIR}/experiment_data/eval/<model-slug>/<RUNTAG>
#                     (subtask names append underneath as <RUNTAG>/<subtask>/)
#   RUNTAG          - leaf directory name (default: moleculenet_default)
#   SUBTASKS        - space-separated list (default: bbbp esol)
#   N_EXAMPLES      - per-sub-task cap (stratified subsample; omit for full)
#   SEED            - random seed for split / sampling (default 0)
#   BOOTSTRAP       - bootstrap resamples for CI (default 200)
#   SPLIT           - "scaffold" (default) or "random"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${MOLECULENET_DIR:?MOLECULENET_DIR must be set}"

TOKENIZER_PATH="${TOKENIZER_PATH:-}"
ARCH="${ARCH:-chemberta2}"
MODALITY="${MODALITY:-compounds}"
DEVICE="${DEVICE:-cuda}"
RUNTAG="${RUNTAG:-moleculenet_default}"
OUTPUT_DIR="${OUTPUT_DIR:-$(compose_eval_output_dir "$MODALITY" "$MODEL_PATH" "$RUNTAG")}"
SUBTASKS="${SUBTASKS:-bbbp esol}"

mkdir -p "$OUTPUT_DIR"

for subtask in $SUBTASKS; do
    task_dir="$MOLECULENET_DIR/$subtask"
    task_out="$OUTPUT_DIR/$subtask"
    echo "Evaluating MoleculeNet/$subtask -> $task_out"

    cmd=("$PYTHON" -m molcrawl.tasks.evaluation.moleculenet
         --model-path "$MODEL_PATH"
         --arch "$ARCH"
         --modality "$MODALITY"
         --device "$DEVICE"
         --subtask "$subtask"
         --task-dir "$task_dir"
         --output-dir "$task_out")
    if [[ -n "$TOKENIZER_PATH" ]]; then
        cmd+=(--tokenizer-path "$TOKENIZER_PATH")
    fi
    if [[ -n "${N_EXAMPLES:-}" ]]; then
        cmd+=(--n-examples "$N_EXAMPLES")
    fi
    if [[ -n "${SEED:-}" ]]; then
        cmd+=(--seed "$SEED")
    fi
    if [[ -n "${BOOTSTRAP:-}" ]]; then
        cmd+=(--bootstrap-samples "$BOOTSTRAP")
    fi
    if [[ -n "${SPLIT:-}" ]]; then
        cmd+=(--split "$SPLIT")
    fi
    if [[ -n "${MAX_EXAMPLES:-}" ]]; then
        cmd+=(--max-examples "$MAX_EXAMPLES")
    fi
    "${cmd[@]}"
done
