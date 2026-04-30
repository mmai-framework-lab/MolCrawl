#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

# Auto-select GPU if not manually specified (large model needs ~20GB)
NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 20

mkdir -p ${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs
LOG_FILE="${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs/molecule_nat_lang-train-large-$(date +%Y-%m-%d_%H-%M-%S).log"
run_training_background "$LOG_FILE" \
    molcrawl/models/gpt2/train.py \
    ./molcrawl/tasks/pretrain/configs/molecule_nat_lang/gpt2_large.py
