#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/genome_sequence/clinvar/logs
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} nohup bash -c '$PYTHON molcrawl/bert/main.py bert/configs/genome_sequence_clinvar_medium.py' > \
    ${LEARNING_SOURCE_DIR}/genome_sequence/clinvar/logs/genome_sequence_clinvar-train-bert-medium-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
