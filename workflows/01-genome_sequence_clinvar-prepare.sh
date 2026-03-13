#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/genome_sequence/clinvar/logs
$PYTHON molcrawl/preparation/preparation_script_genome_sequence.py \
    assets/configs/genome_sequence.yaml \
    --datasets clinvar \
    2>&1 | tee ${LEARNING_SOURCE_DIR}/genome_sequence/clinvar/logs/clinvar-prepare-`date +%Y-%m-%d_%H-%M-%S`.log
