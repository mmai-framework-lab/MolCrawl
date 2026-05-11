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
#   OUTPUT_DIR      - default: experiment_data/eval/moleculenet
#   SUBTASKS        - space-separated list (default: bbbp esol)

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${MOLECULENET_DIR:?MOLECULENET_DIR must be set}"

TOKENIZER_PATH="${TOKENIZER_PATH:-}"
ARCH="${ARCH:-chemberta2}"
MODALITY="${MODALITY:-compounds}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/moleculenet}"
SUBTASKS="${SUBTASKS:-bbbp esol}"

mkdir -p "$OUTPUT_DIR"

for subtask in $SUBTASKS; do
    task_dir="$MOLECULENET_DIR/$subtask"
    task_out="$OUTPUT_DIR/$subtask"
    echo "Evaluating MoleculeNet/$subtask -> $task_out"

    cmd=(python -m molcrawl.tasks.evaluation.moleculenet
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
    "${cmd[@]}"
done
