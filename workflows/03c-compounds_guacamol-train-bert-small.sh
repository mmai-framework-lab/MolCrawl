#!/bin/bash
# Fine-tune the compounds BERT (small) model on GuacaMol.
#
# Prerequisites:
#   - compounds BERT pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/compounds/bert-output/compounds-small/
#   - GuacaMol training_ready_hf_dataset must be prepared via
#       workflows/01-compounds_guacamol-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03c-compounds_guacamol-train-bert-small.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

LOG_DIR="${LEARNING_SOURCE_DIR}/compounds/guacamol/logs"
mkdir -p "${LOG_DIR}"

NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 10

run_training_background "${LOG_DIR}/compounds_guacamol-train-bert-small-$(date +%Y-%m-%d_%H-%M-%S).log" \
    molcrawl/models/bert/main.py \
    bert/configs/compounds_guacamol.py
echo "BERT fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES:-0})."
echo "Logs: ${LOG_DIR}/"
