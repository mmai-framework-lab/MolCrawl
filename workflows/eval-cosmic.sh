#!/usr/bin/env bash
# Phase 3 - COSMIC (Cancer Mutation Census) pathogenicity evaluation.
#
# Required environment:
#   MODEL_PATH      - trained checkpoint (ckpt.pt for gpt2; HF dir for bert)
#   COSMIC_DATA     - CSV produced by molcrawl.tasks.evaluation.cosmic.prepare_csv
#                     (legacy schema: reference_sequence, variant_sequence,
#                     FATHMM_PREDICTION).  See workflows/data/eval-data-cosmic.sh
#                     to download the source CMC TSV first.
#
# Optional:
#   ARCH            - adapter to build (default: gpt2)
#   TOKENIZER_PATH  - required for arch=gpt2 genome
#   OUTPUT_DIR      - default experiment_data/eval/cosmic
#   LABEL_COLUMN    - column to map onto pathogenic/benign (default
#                     FATHMM_PREDICTION; the prep script writes this)
#   CONTEXT_LENGTH  - tokenizer/model context (default 512)
#   N_PER_CLASS     - class-balanced sample size per class (omit for full)
#   STRATIFY_TIER   - "0" disables MUTATION_SIGNIFICANCE_TIER stratification
#                     (default: stratification on)
#   SEED            - random seed for sampling + bootstrap (default 42)
#   BOOTSTRAP_SAMPLES        - bootstrap resamples for 95 % CI (default 200,
#                              set 0 to disable)
#   PREDICTIONS_PREVIEW_COUNT - rows in predictions.txt narrative (default 20)
#   MAX_EXAMPLES    - legacy cap; re-interpreted as N_PER_CLASS = MAX // 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${COSMIC_DATA:?COSMIC_DATA must be set (CSV from prepare_csv)}"

ARCH="${ARCH:-gpt2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/cosmic}"

mkdir -p "$OUTPUT_DIR"

cmd=("$PYTHON" -m molcrawl.tasks.evaluation.cosmic
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality genome_sequence
     --device "$DEVICE"
     --cosmic-data "$COSMIC_DATA"
     --output-dir "$OUTPUT_DIR"
     --label-column "${LABEL_COLUMN:-FATHMM_PREDICTION}"
     --context-length "${CONTEXT_LENGTH:-512}"
     --bootstrap-samples "${BOOTSTRAP_SAMPLES:-200}"
     --predictions-preview-count "${PREDICTIONS_PREVIEW_COUNT:-20}"
     --seed "${SEED:-42}")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ -n "${N_PER_CLASS:-}" ]]; then
    cmd+=(--n-per-class "$N_PER_CLASS")
fi
if [[ "${STRATIFY_TIER:-1}" = "0" ]]; then
    cmd+=(--no-stratify-tier)
fi
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
    cmd+=(--max-examples "$MAX_EXAMPLES")
fi
"${cmd[@]}"
