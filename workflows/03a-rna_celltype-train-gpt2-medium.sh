#!/bin/bash
# Fine-tune the rna GPT-2 (medium) model on the Geneformer cell type
# annotation dataset.
#
# Prerequisites:
#   - rna GPT-2 medium pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/rna/gpt2-output/rna-medium/
#   - CellType training_ready_hf_dataset must be prepared via
#       workflows/01-rna_celltype-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03a-rna_celltype-train-gpt2-medium.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir
auto_select_gpu 20

LOG_DIR="${LEARNING_SOURCE_DIR}/rna/celltype/logs"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} PYTHONUNBUFFERED=1 \
nohup bash -c '$PYTHON molcrawl/gpt2/train.py \
    gpt2/configs/rna/train_gpt2_celltype_medium.py' \
    > "${LOG_DIR}/rna_celltype-train-gpt2-medium-$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

echo "GPT-2 medium fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES})."
echo "Logs: ${LOG_DIR}/"
