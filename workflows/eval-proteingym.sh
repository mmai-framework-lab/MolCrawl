#!/usr/bin/env bash
# Phase 2 - ProteinGym zero-shot mutation effect evaluation.
#
# Required:
#   MODEL_PATH       - trained checkpoint (ckpt.pt for gpt2; HF dir otherwise)
#   PROTEINGYM_DATA  - single per-assay CSV from the ProteinGym release
#
# Optional:
#   ARCH             - adapter to build (default: gpt2)
#   MODALITY         - default: protein_sequence
#   TOKENIZER_PATH   - passed as --tokenizer-path when set
#   OUTPUT_DIR       - default ${LEARNING_SOURCE_DIR}/experiment_data/eval/<model-slug>/<RUNTAG>
#   RUNTAG           - leaf directory name (default: proteingym_default)
#   N_EXAMPLES       - cap on variants scored (omit for full assay)
#   STRATIFY_BIN     - "0" to disable DMS_bin_score stratified sampling
#                      (default: stratification on when DMS_bin_score exists)
#   SEED             - random seed (default 42)
#   BOOTSTRAP        - bootstrap resamples for CI (default 200)
#   MAX_EXAMPLES     - legacy alias for N_EXAMPLES

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${PROTEINGYM_DATA:?PROTEINGYM_DATA must be set}"

ARCH="${ARCH:-gpt2}"
MODALITY="${MODALITY:-protein_sequence}"
DEVICE="${DEVICE:-cuda}"
RUNTAG="${RUNTAG:-proteingym_default}"
OUTPUT_DIR="${OUTPUT_DIR:-$(compose_eval_output_dir "$MODALITY" "$MODEL_PATH" "$RUNTAG")}"

mkdir -p "$OUTPUT_DIR"

cmd=("$PYTHON" -m molcrawl.tasks.evaluation.proteingym
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality "$MODALITY"
     --device "$DEVICE"
     --proteingym-data "$PROTEINGYM_DATA"
     --output-dir "$OUTPUT_DIR")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ -n "${N_EXAMPLES:-}" ]]; then
    cmd+=(--n-examples "$N_EXAMPLES")
fi
if [[ "${STRATIFY_BIN:-1}" = "0" ]]; then
    cmd+=(--no-stratify-bin)
fi
if [[ -n "${SEED:-}" ]]; then
    cmd+=(--seed "$SEED")
fi
if [[ -n "${BOOTSTRAP:-}" ]]; then
    cmd+=(--bootstrap-samples "$BOOTSTRAP")
fi
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
    cmd+=(--max-examples "$MAX_EXAMPLES")
fi
"${cmd[@]}"
