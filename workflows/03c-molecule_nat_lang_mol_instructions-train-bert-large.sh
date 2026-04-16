#!/bin/bash
# Fine-tune the molecule_nat_lang BERT (large) model on Mol-Instructions.
#
# Prerequisites:
#   - molecule_nat_lang BERT pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/molecule_nat_lang/bert-output/molecule_nat_lang-large/
#   - Mol-Instructions training_ready_hf_dataset must be prepared via
#       workflows/01-molecule_nat_lang_mol_instructions-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03c-molecule_nat_lang_mol_instructions-train-bert-large.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

# Use local GPT-2 tokenizer (overridable via env var)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-$PROJECT_ROOT/assets/tokenizers/gpt2}"


LOG_DIR="${LEARNING_SOURCE_DIR}/molecule_nat_lang/mol_instructions/logs"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
nohup bash -c '$PYTHON molcrawl/bert/main.py \
    bert/configs/molecule_nat_lang_mol_instructions_large.py' \
    > "${LOG_DIR}/molecule_nat_lang_mol_instructions-train-bert-large-$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

echo "BERT fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES:-0})."
echo "Logs: ${LOG_DIR}/"
