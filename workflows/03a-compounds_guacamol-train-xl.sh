#!/bin/bash
# Fine-tune the compounds GPT-2 (xl) model on GuacaMol benchmark data.
#
# Prerequisites:
#   - compounds GPT-2 pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/compounds/gpt2-output/compounds-ex-large/
#   - GuacaMol training_ready_hf_dataset must be prepared via
#       workflows/01-compounds_guacamol-prepare.sh and
#       workflows/02-compounds-prepare-gpt2.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03a-compounds_guacamol-train-xl.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir
NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 30

LOG_DIR="${LEARNING_SOURCE_DIR}/compounds/guacamol/logs"
mkdir -p "${LOG_DIR}"

run_training_background "${LOG_DIR}/compounds_guacamol-train-xl-$(date +%Y-%m-%d_%H-%M-%S).log" \
    molcrawl/models/gpt2/train.py \
    molcrawl/tasks/pretrain/configs/compounds/gpt2_guacamol_xl.py

echo "GPT-2 fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES})."
echo "Logs: ${LOG_DIR}/"
