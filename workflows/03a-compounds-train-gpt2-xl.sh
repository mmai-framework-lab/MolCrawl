#!/bin/bash
# Pre-train the compounds GPT-2 (xl) model on organix13 SMILES.
#
# Prerequisites:
#   - Compounds SMILES dataset must be prepared via
#       workflows/01-compounds_prepare.sh
#   - Tokenised GPT-2 dataset must be ready via
#       workflows/02-compounds-prepare-gpt2.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03a-compounds-train-gpt2-xl.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir
auto_select_gpu 30

LOG_DIR="${LEARNING_SOURCE_DIR}/compounds/logs"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} PYTHONUNBUFFERED=1 \
nohup bash -c '$PYTHON molcrawl/gpt2/train.py \
    gpt2/configs/compounds/train_gpt2_xl_config.py' \
    > "${LOG_DIR}/compounds-train-gpt2-xl-$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

echo "GPT-2 pretraining running in background (GPU ${CUDA_VISIBLE_DEVICES})."
echo "Logs: ${LOG_DIR}/"
