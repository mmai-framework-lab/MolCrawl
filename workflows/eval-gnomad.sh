#!/usr/bin/env bash
# Phase 3 - gnomAD allele-frequency correlation.
#
# Required environment:
#   MODEL_PATH      - trained checkpoint (ckpt.pt for gpt2; HF dir otherwise)
#   GNOMAD_DATA     - pre-processed gnomAD CSV (see prepare_csv module)
#
# Optional:
#   ARCH            - adapter to build (default: gpt2)
#   TOKENIZER_PATH  - passed to adapter as --tokenizer-path
#   OUTPUT_DIR      - default ${LEARNING_SOURCE_DIR}/experiment_data/eval/<model-slug>/<RUNTAG>
#   RUNTAG          - leaf directory name (default: gnomad_af_correlation_default)
#   N_PER_BIN       - AF-log-bin stratified sample size (per bin, 6 bins)
#   SEED            - random seed (default 42)
#   BOOTSTRAP       - bootstrap resamples for CI (default 200; 0 disables)
#   MAX_EXAMPLES    - legacy total-row cap; re-interpreted as N_PER_BIN = MAX_EXAMPLES // 6

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${GNOMAD_DATA:?GNOMAD_DATA must be set}"

ARCH="${ARCH:-gpt2}"
DEVICE="${DEVICE:-cuda}"
RUNTAG="${RUNTAG:-gnomad_af_correlation_default}"
OUTPUT_DIR="${OUTPUT_DIR:-$(compose_eval_output_dir genome_sequence "$MODEL_PATH" "$RUNTAG")}"

mkdir -p "$OUTPUT_DIR"

cmd=("$PYTHON" -m molcrawl.tasks.evaluation.gnomad_af_correlation
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality genome_sequence
     --device "$DEVICE"
     --gnomad-data "$GNOMAD_DATA"
     --output-dir "$OUTPUT_DIR")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ -n "${N_PER_BIN:-}" ]]; then
    cmd+=(--n-per-bin "$N_PER_BIN")
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
