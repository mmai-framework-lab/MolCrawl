#!/usr/bin/env bash
# Smoke workflow for the new evaluation framework (Phase 0).
#
# Runs the ClinVar pilot evaluator end-to-end on a small sample to make
# sure BaseEvaluator / ModelAdapter / ReportWriter wire up correctly.
#
# Required environment:
#   MODEL_PATH      - path to a trained GPT-2 genome_sequence checkpoint
#   TOKENIZER_PATH  - path to the SentencePiece tokenizer used for training
#   CLINVAR_DATA    - path to a pre-prepared ClinVar CSV/TSV/JSON
#
# Optional:
#   OUTPUT_DIR      - directory to write metrics.json / REPORT.md
#                     (default: ${LEARNING_SOURCE_DIR}/experiment_data/eval/_smoke/<model-slug>/clinvar_smoke)
#   RUNTAG          - leaf directory name (default: clinvar_smoke)
#   DEVICE          - torch device string (default: cuda)
#   MAX_EXAMPLES    - cap evaluated variants (default: 16)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${TOKENIZER_PATH:?TOKENIZER_PATH must be set}"
: "${CLINVAR_DATA:?CLINVAR_DATA must be set}"

RUNTAG="${RUNTAG:-clinvar_smoke}"
# Smoke runs land under the ``_smoke/`` bucket of the new layout.
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    : "${LEARNING_SOURCE_DIR:?LEARNING_SOURCE_DIR must be set for smoke output}"
    _SLUG="$(derive_model_slug genome_sequence "$MODEL_PATH")"
    if [[ -n "$_SLUG" ]]; then
        OUTPUT_DIR="${LEARNING_SOURCE_DIR%/}/experiment_data/eval/_smoke/${_SLUG}/${RUNTAG}"
    else
        OUTPUT_DIR="${LEARNING_SOURCE_DIR%/}/experiment_data/eval/_smoke/${RUNTAG}"
    fi
fi
DEVICE="${DEVICE:-cuda}"
MAX_EXAMPLES="${MAX_EXAMPLES:-16}"

mkdir -p "$OUTPUT_DIR"

"$PYTHON" -m molcrawl.tasks.evaluation.clinvar \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --clinvar-data "$CLINVAR_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --arch gpt2 \
    --modality genome_sequence \
    --device "$DEVICE" \
    --max-examples "$MAX_EXAMPLES"

echo "Smoke evaluation complete: $OUTPUT_DIR"
