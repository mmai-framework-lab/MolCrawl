#!/bin/bash
# Tokenise GuacaMol benchmark SMILES and save as HuggingFace Dataset for GPT-2.
#
# Prerequisites:
#   - GuacaMol benchmark files must be downloaded via
#       workflows/01-compounds_guacamol-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/02-compounds-prepare-gpt2.sh

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

mkdir -p ${LEARNING_SOURCE_DIR}/compounds/logs/
nohup $PYTHON molcrawl/compounds/dataset/prepare_gpt2.py assets/configs/compounds.yaml \
    > ${LEARNING_SOURCE_DIR}/compounds/logs/compounds-prepare-gpt2-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &

echo "GuacaMol dataset preparation running in background. Logs: ${LEARNING_SOURCE_DIR}/compounds/logs/"
