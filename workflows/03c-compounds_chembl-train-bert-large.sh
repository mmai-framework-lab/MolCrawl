#!/bin/bash
# Fine-tune the compounds BERT (large) model on ChEMBL.
#
# Prerequisites:
#   - compounds BERT pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/compounds/bert-output/compounds-large/
#   - ChEMBL training_ready_hf_dataset must be prepared via
#       workflows/01-compounds_chembl-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03c-compounds_chembl-train-bert-large.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

LOG_DIR="${LEARNING_SOURCE_DIR}/compounds/chembl/logs"
mkdir -p "${LOG_DIR}"

NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 40

run_training_background "${LOG_DIR}/compounds_chembl-train-bert-large-$(date +%Y-%m-%d_%H-%M-%S).log" \
    molcrawl/models/bert/main.py \
    bert/configs/compounds_chembl_large.py

echo "BERT fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES:-0})."
echo "Logs: ${LOG_DIR}/"
