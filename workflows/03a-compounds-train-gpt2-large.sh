#!/bin/bash
# Pre-train the compounds GPT-2 (large) model on organix13 SMILES.
#
# Prerequisites:
#   - Compounds SMILES dataset must be prepared via
#       workflows/01-compounds_prepare.sh
#   - Tokenised GPT-2 dataset must be ready via
#       workflows/02-compounds-prepare-gpt2.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03a-compounds-train-gpt2-large.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir
NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 20

LOG_DIR="${LEARNING_SOURCE_DIR}/compounds/logs"
mkdir -p "${LOG_DIR}"

run_training_background "${LOG_DIR}/compounds-train-gpt2-large-$(date +%Y-%m-%d_%H-%M-%S).log" \
    molcrawl/gpt2/train.py \
    gpt2/configs/compounds/train_gpt2_large_config.py

echo "GPT-2 pretraining running in background (GPU ${CUDA_VISIBLE_DEVICES})."
echo "Logs: ${LOG_DIR}/"
