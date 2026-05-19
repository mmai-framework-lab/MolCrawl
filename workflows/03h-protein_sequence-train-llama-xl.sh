#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 10

mkdir -p ${LEARNING_SOURCE_DIR}/protein_sequence/logs
LOG_FILE="${LEARNING_SOURCE_DIR}/protein_sequence/logs/protein_sequence-train-llama-xl-$(date +%Y-%m-%d_%H-%M-%S).log"
run_training_background "$LOG_FILE" \
    molcrawl/models/llama/train.py \
    ./molcrawl/tasks/pretrain/configs/protein_sequence/llama_xl.py
