#!/usr/bin/env bash
# Phase 5 - ChemLLMBench evaluation across all nine sub-tasks.

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${CHEMLLMBENCH_DIR:?CHEMLLMBENCH_DIR must be set}"

ARCH="${ARCH:-gpt2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/chemllmbench}"
SUBTASKS="${SUBTASKS:-}"

mkdir -p "$OUTPUT_DIR"

if [[ -z "$SUBTASKS" ]]; then
    SUBTASKS=$(python -c "from molcrawl.tasks.evaluation.chemllmbench.data_preparation import TASKS; print(' '.join(TASKS))")
fi

for subtask in $SUBTASKS; do
    jsonl="$CHEMLLMBENCH_DIR/$subtask.jsonl"
    if [[ ! -f "$jsonl" ]]; then
        echo "Skipping missing $jsonl" >&2
        continue
    fi
    cmd=(python -m molcrawl.tasks.evaluation.chemllmbench
         --model-path "$MODEL_PATH"
         --arch "$ARCH"
         --modality molecule_nat_lang
         --device "$DEVICE"
         --task "$subtask"
         --jsonl-path "$jsonl"
         --output-dir "$OUTPUT_DIR/$subtask")
    if [[ -n "${TOKENIZER_PATH:-}" ]]; then
        cmd+=(--tokenizer-path "$TOKENIZER_PATH")
    fi
    "${cmd[@]}"
done
