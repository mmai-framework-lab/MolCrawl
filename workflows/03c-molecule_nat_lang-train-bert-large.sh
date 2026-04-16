#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} nohup bash -c '$PYTHON molcrawl/bert/main.py bert/configs/molecule_nat_lang_large.py' > \
    ${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs/molecule_nat_lang-train-bert-large-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
