#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/genome_sequence/logs
nohup bash -c 'python gpt2/train.py ./gpt2/configs/genome_sequence/train_gpt2_config.py' --use_wandb=True --wandb_project="genome-sequence" > \
    ${LEARNING_SOURCE_DIR}/genome_sequence/logs/genome_sequence-train-small-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &