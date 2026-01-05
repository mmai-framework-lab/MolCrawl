#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/molecule_nl/logs
nohup bash -c 'python bert/main.py bert/configs/molecule_nl.py' > \
    ${LEARNING_SOURCE_DIR}/molecule_nl/logs/molecule_nl-train-bert-small-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
