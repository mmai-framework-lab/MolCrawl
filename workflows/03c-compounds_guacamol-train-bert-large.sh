#!/bin/bash
# Fine-tune the compounds BERT (large) model on GuacaMol.
#
# Prerequisites:
#   - compounds BERT pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/compounds/bert-output/compounds-large/
#   - GuacaMol training_ready_hf_dataset must be prepared via
#       workflows/01-compounds_guacamol-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03c-compounds_guacamol-train-bert-large.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

LOG_DIR="${LEARNING_SOURCE_DIR}/compounds/guacamol/logs"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
nohup bash -c '$PYTHON molcrawl/bert/main.py \
    bert/configs/compounds_guacamol_large.py' \
    > "${LOG_DIR}/compounds_guacamol-train-bert-large-$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &
echo "BERT fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES:-0})."
echo "Logs: ${LOG_DIR}/"
