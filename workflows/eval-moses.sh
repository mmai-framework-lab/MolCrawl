#!/usr/bin/env bash
# Phase 1 - MOSES generation-quality evaluation for compound decoders.
#
# Required environment:
#   MODEL_PATH       - path to a GPT-2 compound checkpoint
#   TOKENIZER_PATH   - SentencePiece tokenizer / SMILES vocab path
#   MOSES_DIR        - directory containing train.csv / test.csv /
#                      test_scaffolds.csv / manifest.json
#
# Optional:
#   NUM_SAMPLES               - default 30000
#   TEMPERATURE               - default 1.0
#   TOP_K                     - default unset (greedy filter off)
#   MAX_NEW_TOKENS            - default 128
#   OUTPUT_DIR                - default experiment_data/eval/moses
#   SEED                      - torch sampling seed (default 42)
#   BOOTSTRAP                 - bootstrap resamples for CI (default 100; 0 disables)
#   PREDICTIONS_PREVIEW_COUNT - narrative size (default 30; 0 disables)
#   REFERENCE_LIMIT           - cap train reference (faster smoke)
#   TEST_LIMIT                - cap test reference
#   SCAFFOLDS_LIMIT           - cap test_scaffolds reference
#   DISABLE_EXTENDED          - "1" to skip optional FCD / SNN metrics
#   NO_SCAFFOLDS_NOVELTY      - "1" to skip novelty vs test_scaffolds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${TOKENIZER_PATH:?TOKENIZER_PATH must be set}"
: "${MOSES_DIR:?MOSES_DIR must be set}"

NUM_SAMPLES="${NUM_SAMPLES:-30000}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/moses}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

mkdir -p "$OUTPUT_DIR"

cmd=("$PYTHON" -m molcrawl.tasks.evaluation.moses
     --model-path "$MODEL_PATH"
     --tokenizer-path "$TOKENIZER_PATH"
     --arch gpt2
     --modality compounds
     --device "$DEVICE"
     --reference-dir "$MOSES_DIR"
     --output-dir "$OUTPUT_DIR"
     --num-samples "$NUM_SAMPLES"
     --temperature "$TEMPERATURE"
     --max-new-tokens "$MAX_NEW_TOKENS")
if [[ -n "${TOP_K:-}" ]]; then
    cmd+=(--top-k "$TOP_K")
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
if [[ -n "${REFERENCE_LIMIT:-}" ]]; then
    cmd+=(--reference-limit "$REFERENCE_LIMIT")
fi
if [[ -n "${TEST_LIMIT:-}" ]]; then
    cmd+=(--test-limit "$TEST_LIMIT")
fi
if [[ -n "${SCAFFOLDS_LIMIT:-}" ]]; then
    cmd+=(--scaffolds-limit "$SCAFFOLDS_LIMIT")
fi
if [[ "${DISABLE_EXTENDED:-0}" = "1" ]]; then
    cmd+=(--disable-extended)
fi
if [[ "${NO_SCAFFOLDS_NOVELTY:-0}" = "1" ]]; then
    cmd+=(--no-scaffolds-novelty)
fi
"${cmd[@]}"
