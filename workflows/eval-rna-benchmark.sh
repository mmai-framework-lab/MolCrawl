#!/usr/bin/env bash
# Phase 4 - rna_benchmark evaluation.
#
# Required:
#   MODEL_PATH       - encoder MLM checkpoint (bert / rnaformer)
#   RNA_JSONL        - per-tissue cell JSONL produced by
#                      `python -m molcrawl.tasks.evaluation.rna_benchmark.prepare_jsonl`
#
# Optional:
#   ARCH                       - default bert
#   TOKENIZER_PATH             - default uses arch+modality fallback
#   OUTPUT_DIR                 - default experiment_data/eval/rna_benchmark
#   DATASETS                   - space-separated list of tissue tags to keep
#   CELLS_PER_GROUP            - reproducibly subsample to this many cells/tissue
#   SEED                       - default 42
#   BOOTSTRAP                  - default 100 (0 disables CI)
#   PREDICTIONS_PREVIEW_COUNT  - default 6

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${RNA_JSONL:?RNA_JSONL must be set}"

ARCH="${ARCH:-bert}"
DEVICE="${DEVICE:-cuda}"
RUNTAG="${RUNTAG:-rna_benchmark_default}"
OUTPUT_DIR="${OUTPUT_DIR:-$(compose_eval_output_dir rna "$MODEL_PATH" "$RUNTAG")}"

mkdir -p "$OUTPUT_DIR"

cmd=("$PYTHON" -m molcrawl.tasks.evaluation.rna_benchmark
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality rna
     --device "$DEVICE"
     --rna-jsonl "$RNA_JSONL"
     --output-dir "$OUTPUT_DIR")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ -n "${DATASETS:-}" ]]; then
    cmd+=(--datasets $DATASETS)
fi
if [[ -n "${CELLS_PER_GROUP:-}" ]]; then
    cmd+=(--cells-per-group "$CELLS_PER_GROUP")
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
