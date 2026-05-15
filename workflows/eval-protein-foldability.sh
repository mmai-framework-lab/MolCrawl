#!/usr/bin/env bash
# Phase 2 - structure-free foldability proxies.
#
# Required:
#   MODEL_PATH       - path to a protein decoder checkpoint (gpt2)
#   REFERENCE_FASTA  - FASTA used as the reference corpus
#                      (e.g. eval/protein_foldability/pdb_seqres.txt)
#
# Optional:
#   ARCH                       - default gpt2 (only gpt2 supports generate)
#   TOKENIZER_PATH             - passed as --tokenizer-path when set
#   OUTPUT_DIR                 - default ${LEARNING_SOURCE_DIR}/experiment_data/eval/<model-slug>/<RUNTAG>
#   RUNTAG                     - leaf directory name (default: protein_foldability_default)
#   NUM_SAMPLES                - default 200
#   TEMPERATURE                - default 1.0
#   TOP_K                      - default unset
#   MAX_NEW_TOKENS             - default 256
#   SEED                       - torch sampling seed (default 42)
#   BOOTSTRAP                  - bootstrap resamples (default 100; 0 disables)
#   PREDICTIONS_PREVIEW_COUNT  - narrative preview size (default 30)
#   FOLDABLE_MIN_LENGTH        - novel-long quadrant threshold (default 50 aa)
#   MAX_REF_FOR_AA             - cap reference for AA-distribution computation
#                                (membership still uses full corpus). Useful
#                                for the ~1.1 M RCSB pdb_seqres FASTA.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${REFERENCE_FASTA:?REFERENCE_FASTA must be set}"

ARCH="${ARCH:-gpt2}"
DEVICE="${DEVICE:-cuda}"
RUNTAG="${RUNTAG:-protein_foldability_default}"
OUTPUT_DIR="${OUTPUT_DIR:-$(compose_eval_output_dir protein_sequence "$MODEL_PATH" "$RUNTAG")}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"

mkdir -p "$OUTPUT_DIR"

cmd=("$PYTHON" -m molcrawl.tasks.evaluation.protein_foldability
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality protein_sequence
     --device "$DEVICE"
     --reference-fasta "$REFERENCE_FASTA"
     --output-dir "$OUTPUT_DIR"
     --num-samples "$NUM_SAMPLES")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ -n "${TEMPERATURE:-}" ]]; then
    cmd+=(--temperature "$TEMPERATURE")
fi
if [[ -n "${TOP_K:-}" ]]; then
    cmd+=(--top-k "$TOP_K")
fi
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
    cmd+=(--max-new-tokens "$MAX_NEW_TOKENS")
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
if [[ -n "${FOLDABLE_MIN_LENGTH:-}" ]]; then
    cmd+=(--foldable-min-length "$FOLDABLE_MIN_LENGTH")
fi
if [[ -n "${MAX_REF_FOR_AA:-}" ]]; then
    cmd+=(--max-ref-for-aa "$MAX_REF_FOR_AA")
fi
"${cmd[@]}"
