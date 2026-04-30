#!/usr/bin/env bash
# Phase 1 - MOSES generation-quality evaluation for compound decoders.
#
# Required environment:
#   MODEL_PATH      - path to a GPT-2 compound checkpoint
#   TOKENIZER_PATH  - SentencePiece tokenizer
#   MOSES_DIR       - directory containing train.csv / test.csv / manifest.json
#
# Optional:
#   NUM_SAMPLES     - default 30000
#   TEMPERATURE     - default 1.0
#   OUTPUT_DIR      - default experiment_data/eval/moses

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${TOKENIZER_PATH:?TOKENIZER_PATH must be set}"
: "${MOSES_DIR:?MOSES_DIR must be set}"

NUM_SAMPLES="${NUM_SAMPLES:-30000}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/moses}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

mkdir -p "$OUTPUT_DIR"

python -m molcrawl.tasks.evaluation.moses \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --arch gpt2 \
    --modality compounds \
    --device "$DEVICE" \
    --reference-dir "$MOSES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --max-new-tokens "$MAX_NEW_TOKENS"
