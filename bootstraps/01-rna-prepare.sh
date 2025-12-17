#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/rna/logs/
nohup python scripts/preparation/preparation_script_rna.py assets/configs/rna.yaml \
> ${LEARNING_SOURCE_DIR}/rna/logs/rna-preparation-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &