#!/bin/bash
# Fine-tune the protein_sequence BERT (large) model on ProteinGym DMS data.
#
# Prerequisites:
#   - protein_sequence BERT pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/protein_sequence/bert-output/protein_sequence-large/
#   - ProteinGym training_ready_hf_dataset must be prepared via
#       workflows/01-protein_sequence_proteingym-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03c-protein_sequence_proteingym-train-bert-large.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

LOG_DIR="${LEARNING_SOURCE_DIR}/protein_sequence/proteingym/logs"
mkdir -p "${LOG_DIR}"

NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 40

run_training_background "${LOG_DIR}/protein_sequence_proteingym-train-bert-large-$(date +%Y-%m-%d_%H-%M-%S).log" \
    molcrawl/bert/main.py \
    bert/configs/protein_sequence_proteingym_large.py

echo "BERT fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES:-0})."
echo "Logs: ${LOG_DIR}/"
