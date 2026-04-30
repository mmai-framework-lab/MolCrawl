#!/bin/bash
# Integrate tokenised Organix-13 compound datasets and save as HuggingFace Dataset
# for GPT-2 training.
#
# Prerequisites:
#   - Tokenised parquet files must exist (run workflows/01-compounds_prepare.sh first)
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/02-compounds_organix13-prepare-gpt2.sh

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

mkdir -p ${LEARNING_SOURCE_DIR}/compounds/logs/
nohup $PYTHON molcrawl/data/compounds/dataset/prepare_gpt2_organix13.py assets/configs/compounds.yaml \
    > ${LEARNING_SOURCE_DIR}/compounds/logs/compounds-organix13-prepare-gpt2-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &

echo "Organix-13 dataset preparation running in background. Logs: ${LEARNING_SOURCE_DIR}/compounds/logs/"
