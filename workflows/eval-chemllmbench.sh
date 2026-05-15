#!/usr/bin/env bash
# Phase 5 - ChemLLMBench evaluation across all nine sub-tasks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${CHEMLLMBENCH_DIR:?CHEMLLMBENCH_DIR must be set}"

ARCH="${ARCH:-gpt2}"
DEVICE="${DEVICE:-cuda}"
RUNTAG="${RUNTAG:-chemllmbench_default}"
OUTPUT_DIR="${OUTPUT_DIR:-$(compose_eval_output_dir molecule_nat_lang "$MODEL_PATH" "$RUNTAG")}"
SUBTASKS="${SUBTASKS:-}"

mkdir -p "$OUTPUT_DIR"

if [[ -z "$SUBTASKS" ]]; then
    SUBTASKS=$("$PYTHON" -c "from molcrawl.tasks.evaluation.chemllmbench.data_preparation import TASKS; print(' '.join(TASKS))")
fi

for subtask in $SUBTASKS; do
    jsonl="$CHEMLLMBENCH_DIR/$subtask.jsonl"
    if [[ ! -f "$jsonl" ]]; then
        echo "Skipping missing $jsonl" >&2
        continue
    fi
    cmd=("$PYTHON" -m molcrawl.tasks.evaluation.chemllmbench
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
    if [[ -n "${MAX_EXAMPLES:-}" ]]; then
        cmd+=(--max-examples "$MAX_EXAMPLES")
    fi
    if [[ -n "${SEED:-}" ]]; then
        cmd+=(--seed "$SEED")
    fi
    if [[ -n "${PREDICTIONS_PREVIEW_COUNT:-}" ]]; then
        cmd+=(--predictions-preview-count "$PREDICTIONS_PREVIEW_COUNT")
    fi
    if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
        cmd+=(--max-new-tokens "$MAX_NEW_TOKENS")
    fi
    if [[ -n "${TEMPERATURE:-}" ]]; then
        cmd+=(--temperature "$TEMPERATURE")
    fi
    "${cmd[@]}"
done
