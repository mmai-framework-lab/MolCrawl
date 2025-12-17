#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/compounds/logs
nohup bash -c 'python gpt2/train.py gpt2/configs/compounds/train_gpt2_medium_config.py' > \
    ${LEARNING_SOURCE_DIR}/compounds/logs/compounds-train-medium-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &