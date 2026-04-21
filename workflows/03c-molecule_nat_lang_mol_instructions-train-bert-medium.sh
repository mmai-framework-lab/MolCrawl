#!/bin/bash
# Fine-tune the molecule_nat_lang BERT (medium) model on Mol-Instructions.
#
# Prerequisites:
#   - molecule_nat_lang BERT pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/molecule_nat_lang/bert-output/molecule_nat_lang-medium/
#   - Mol-Instructions training_ready_hf_dataset must be prepared via
#       workflows/01-molecule_nat_lang_mol_instructions-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03c-molecule_nat_lang_mol_instructions-train-bert-medium.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

# Use local GPT-2 tokenizer (overridable via env var)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# Offline GPT-2 tokenizer: export only if the directory actually exists.
# Otherwise tokenizer.py falls back to "gpt2" via the HF cache.
if [ -z "${GPT2_TOKENIZER_DIR:-}" ] && [ -d "$PROJECT_ROOT/assets/tokenizers/gpt2" ]; then
    export GPT2_TOKENIZER_DIR="$PROJECT_ROOT/assets/tokenizers/gpt2"
fi


LOG_DIR="${LEARNING_SOURCE_DIR}/molecule_nat_lang/mol_instructions/logs"
mkdir -p "${LOG_DIR}"

NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 20

run_training_background "${LOG_DIR}/molecule_nat_lang_mol_instructions-train-bert-medium-$(date +%Y-%m-%d_%H-%M-%S).log" \
    molcrawl/bert/main.py \
    bert/configs/molecule_nat_lang_mol_instructions_medium.py

echo "BERT fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES:-0})."
echo "Logs: ${LOG_DIR}/"
