#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/protein_sequence/logs/
nohup $PYTHON molcrawl/preparation/preparation_script_protein_sequence.py assets/configs/protein_sequence.yaml \
> ${LEARNING_SOURCE_DIR}/protein_sequence/logs/protein-sequence-preparation-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &
