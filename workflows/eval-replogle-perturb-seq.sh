#!/usr/bin/env bash
# Phase 4 - Replogle Perturb-seq evaluation.
#
# Required:
#   MODEL_PATH       - encoder MLM checkpoint (rnaformer / bert rna)
#   REPLOGLE_DATA    - path to the (perturbation, baseline, perturbed) CSV
#                      built by workflows/data/eval-data-replogle-perturb-seq.sh
#
# Optional:
#   ARCH                       - default rnaformer
#   TOKENIZER_PATH             - default uses arch+modality fallback
#   OUTPUT_DIR                 - default experiment_data/eval/replogle_perturb_seq
#   TEST_FRACTION              - default 0.2 (perturbation-disjoint split)
#   MAX_EXAMPLES               - delta-strength-aware subsample size
#   SEED                       - default 0
#   BOOTSTRAP                  - default 100 (0 disables CI)
#   PREDICTIONS_PREVIEW_COUNT  - default 16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${REPLOGLE_DATA:?REPLOGLE_DATA must be set}"

ARCH="${ARCH:-rnaformer}"
DEVICE="${DEVICE:-cuda}"
RUNTAG="${RUNTAG:-replogle_perturb_seq_default}"
OUTPUT_DIR="${OUTPUT_DIR:-$(compose_eval_output_dir rna "$MODEL_PATH" "$RUNTAG")}"

mkdir -p "$OUTPUT_DIR"

cmd=("$PYTHON" -m molcrawl.tasks.evaluation.replogle_perturb_seq
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality rna
     --device "$DEVICE"
     --replogle-data "$REPLOGLE_DATA"
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
