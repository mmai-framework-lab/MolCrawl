#!/bin/bash
# Fine-tune the protein_sequence BERT (small) model on ProteinGym DMS data.
#
# Prerequisites:
#   - protein_sequence BERT pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/protein_sequence/bert-output/protein_sequence-small/
#   - ProteinGym training_ready_hf_dataset must be prepared via
#       workflows/01-protein_sequence_proteingym-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03c-protein_sequence_proteingym-train-bert-small.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

LOG_DIR="${LEARNING_SOURCE_DIR}/protein_sequence/proteingym/logs"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 \
nohup bash -c '$PYTHON molcrawl/bert/main.py \
    bert/configs/protein_sequence_proteingym.py' \
    > "${LOG_DIR}/protein_sequence_proteingym-train-bert-small-$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

echo "BERT fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES:-0})."
echo "Logs: ${LOG_DIR}/"
