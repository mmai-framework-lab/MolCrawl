#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

# Auto-select GPU if not manually specified (xl model needs ~30GB)
auto_select_gpu 30

mkdir -p ${LEARNING_SOURCE_DIR}/genome_sequence/logs
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nohup bash -c 'python gpt2/train.py ./gpt2/configs/genome_sequence/train_gpt2_xl_config.py' > \
    ${LEARNING_SOURCE_DIR}/genome_sequence/logs/genome_sequence-train-xl-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
