#!/usr/bin/env bash
# Phase 5 - molecule_nat_lang pair scoring.
#
# Required:
#   MODEL_PATH       - GPT-2 / BERT checkpoint trained on molecule_nat_lang corpus
#   PAIRS_CSV        - CSV with smiles + caption columns
#                      (build via `python -m molcrawl.tasks.evaluation.molecule_nat_lang.prepare_pairs`)
#
# Optional:
#   ARCH                       - default gpt2
#   TOKENIZER_PATH             - default uses arch+modality fallback
#   OUTPUT_DIR                 - default experiment_data/eval/molecule_nat_lang
#   SMILES_COLUMN              - default 'smiles'
#   CAPTION_COLUMN             - default 'caption'
#   TEMPLATE                   - default '{caption}\n{smiles}'
#   MAX_EXAMPLES               - combined-length-stratified subsample size
#   SEED                       - default 42
#   BOOTSTRAP                  - default 100 (0 disables CI)
#   PREDICTIONS_PREVIEW_COUNT  - default 20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${PAIRS_CSV:?PAIRS_CSV must be set}"

ARCH="${ARCH:-gpt2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval/molecule_nat_lang}"

mkdir -p "$OUTPUT_DIR"

cmd=("$PYTHON" -m molcrawl.tasks.evaluation.molecule_nat_lang
     --model-path "$MODEL_PATH"
     --arch "$ARCH"
     --modality molecule_nat_lang
     --device "$DEVICE"
     --pairs-csv "$PAIRS_CSV"
     --output-dir "$OUTPUT_DIR")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    cmd+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ -n "${SMILES_COLUMN:-}" ]]; then
    cmd+=(--smiles-column "$SMILES_COLUMN")
fi
if [[ -n "${CAPTION_COLUMN:-}" ]]; then
    cmd+=(--caption-column "$CAPTION_COLUMN")
fi
if [[ -n "${TEMPLATE:-}" ]]; then
    cmd+=(--template "$TEMPLATE")
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
