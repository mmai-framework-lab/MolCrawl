#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

# Auto-select GPU if not manually specified (large model needs ~20GB)
auto_select_gpu 20

mkdir -p ${LEARNING_SOURCE_DIR}/compounds/logs
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nohup bash -c 'python src/gpt2/train.py gpt2/configs/compounds/train_gpt2_large_config.py' > \
    ${LEARNING_SOURCE_DIR}/compounds/logs/compounds-train-large-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &