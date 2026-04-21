#!/bin/bash
# Fine-tune the molecule_nat_lang GPT-2 (small) model on Mol-Instructions.
#
# Prerequisites:
#   - molecule_nat_lang GPT-2 pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/molecule_nat_lang/gpt2-output/molecule_nat_lang-small/
#   - Mol-Instructions training_ready_hf_dataset must be prepared via
#       workflows/01-molecule_nat_lang_mol_instructions-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03a-molecule_nat_lang_mol_instructions-train-small.sh

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

NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 10

LOG_DIR="${LEARNING_SOURCE_DIR}/molecule_nat_lang/mol_instructions/logs"
mkdir -p "${LOG_DIR}"

run_training_background "${LOG_DIR}/molecule_nat_lang_mol_instructions-train-small-$(date +%Y-%m-%d_%H-%M-%S).log" \
    molcrawl/gpt2/train.py \
    gpt2/configs/molecule_nat_lang/train_gpt2_mol_instructions_small.py

echo "GPT-2 fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES})."
echo "Logs: ${LOG_DIR}/"
