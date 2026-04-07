#!/bin/bash
# Fine-tune the molecule_nat_lang GPT-2 (medium) model on Mol-Instructions.
#
# Prerequisites:
#   - molecule_nat_lang GPT-2 medium pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/molecule_nat_lang/gpt2-output/molecule_nat_lang-medium/
#   - Mol-Instructions training_ready_hf_dataset must be prepared via
#       workflows/01-molecule_nat_lang_mol_instructions-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03a-molecule_nat_lang_mol_instructions-train-medium.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

# Use local GPT-2 tokenizer (overridable via env var)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-$PROJECT_ROOT/assets/tokenizers/gpt2}"

auto_select_gpu 20

LOG_DIR="${LEARNING_SOURCE_DIR}/molecule_nat_lang/mol_instructions/logs"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} PYTHONUNBUFFERED=1 \
nohup bash -c '$PYTHON molcrawl/gpt2/train.py \
    gpt2/configs/molecule_nat_lang/train_gpt2_mol_instructions_medium.py' \
    > "${LOG_DIR}/molecule_nat_lang_mol_instructions-train-medium-$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

echo "GPT-2 medium Mol-Instructions fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES})."
echo "Logs: ${LOG_DIR}/"
