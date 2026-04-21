#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

# Use local GPT-2 tokenizer (overridable via env var)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-$PROJECT_ROOT/assets/tokenizers/gpt2}"


# Auto-select GPU if not manually specified (small model needs ~10GB)
NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 10

mkdir -p ${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs
LOG_FILE="${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs/molecule_nat_lang-train-small-$(date +%Y-%m-%d_%H-%M-%S).log"
run_training_background "$LOG_FILE" \
    molcrawl/gpt2/train.py \
    ./gpt2/configs/molecule_nat_lang/train_gpt2_small_config.py
