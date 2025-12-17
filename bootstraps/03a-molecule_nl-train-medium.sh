#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/molecule_nl/logs
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python gpt2/train.py ./gpt2/configs/molecule_nl/train_gpt2_medium_config.py' > \
    ${LEARNING_SOURCE_DIR}/molecule_nl/logs/molecule_nl-train-medium-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &