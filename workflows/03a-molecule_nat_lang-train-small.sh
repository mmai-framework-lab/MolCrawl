#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

# Use local GPT-2 tokenizer (overridable via env var)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# Offline GPT-2 tokenizer: export only if the directory actually exists.
# Otherwise tokenizer.py falls back to "gpt2" via the HF cache.
if [ -z "${GPT2_TOKENIZER_DIR:-}" ] && [ -d "$PROJECT_ROOT/assets/tokenizers/gpt2" ]; then
    export GPT2_TOKENIZER_DIR="$PROJECT_ROOT/assets/tokenizers/gpt2"
fi


# Auto-select GPU if not manually specified (small model needs ~10GB)
NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 10

mkdir -p ${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs
LOG_FILE="${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs/molecule_nat_lang-train-small-$(date +%Y-%m-%d_%H-%M-%S).log"
run_training_background "$LOG_FILE" \
    molcrawl/models/gpt2/train.py \
    ./molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_small.py
