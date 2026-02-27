#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/genome_sequence/logs/
nohup bash -c '$PYTHON molcrawl/genome_sequence/dataset/prepare_gpt2.py assets/configs/genome_sequence.yaml' > \
    ${LEARNING_SOURCE_DIR}/genome_sequence/logs/genome_sequence-prepare-gpt2-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
