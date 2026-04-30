#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs
NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 40

LOG_FILE="${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs/molecule_nat_lang-train-bert-large-$(date +%Y-%m-%d_%H-%M-%S).log"
run_training_background "$LOG_FILE" \
    molcrawl/models/bert/main.py \
    molcrawl/tasks/pretrain/configs/molecule_nat_lang/bert_large.py