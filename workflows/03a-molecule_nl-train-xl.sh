#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

# Auto-select GPU if not manually specified (xl model needs ~30GB)
auto_select_gpu 30

mkdir -p ${LEARNING_SOURCE_DIR}/molecule_nl/logs
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nohup bash -c 'python src/gpt2/train.py ./gpt2/configs/molecule_nl/train_gpt2_xl_config.py' > \
    ${LEARNING_SOURCE_DIR}/molecule_nl/logs/molecule_nl-train-xl-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &