#!/bin/bash
# Fine-tune the compounds BERT (medium) model on ChEMBL.
#
# Prerequisites:
#   - compounds BERT pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/compounds/bert-output/compounds-medium/
#   - ChEMBL training_ready_hf_dataset must be prepared via
#       workflows/01-compounds_chembl-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03c-compounds_chembl-train-bert-medium.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

LOG_DIR="${LEARNING_SOURCE_DIR}/compounds/chembl/logs"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
nohup bash -c '$PYTHON molcrawl/bert/main.py \
    bert/configs/compounds_chembl_medium.py' \
    > "${LOG_DIR}/compounds_chembl-train-bert-medium-$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

echo "BERT fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES:-0})."
echo "Logs: ${LOG_DIR}/"
