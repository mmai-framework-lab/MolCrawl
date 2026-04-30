#!/bin/bash
# Fine-tune the rna BERT (medium) model on the Geneformer cell type
# annotation dataset.
#
# Prerequisites:
#   - rna BERT pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/rna/bert-output/rna-medium/
#   - CellType training_ready_hf_dataset must be prepared via
#       workflows/01-rna_celltype-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03c-rna_celltype-train-bert-medium.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

LOG_DIR="${LEARNING_SOURCE_DIR}/rna/celltype/logs"
mkdir -p "${LOG_DIR}"

NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 20

run_training_background "${LOG_DIR}/rna_celltype-train-bert-medium-$(date +%Y-%m-%d_%H-%M-%S).log" \
    molcrawl/models/bert/main.py \
    molcrawl/tasks/pretrain/configs/rna/bert_celltype_medium.py

echo "BERT fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES:-0})."
echo "Logs: ${LOG_DIR}/"
