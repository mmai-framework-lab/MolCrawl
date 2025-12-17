#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/rna/logs
nohup bash -c 'python gpt2/train.py gpt2/configs/rna/train_gpt2_config_yigarashi.py' > \
    ${LEARNING_SOURCE_DIR}/rna/logs/rna-yigarashi-train-small-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &