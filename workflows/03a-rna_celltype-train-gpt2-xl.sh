#!/bin/bash
# Fine-tune the rna GPT-2 (xl) model on the Geneformer cell type
# annotation dataset.
#
# Prerequisites:
#   - rna GPT-2 xl pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/rna/gpt2-output/rna-ex-large/
#   - CellType training_ready_hf_dataset must be prepared via
#       workflows/01-rna_celltype-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03a-rna_celltype-train-gpt2-xl.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir
NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 35

LOG_DIR="${LEARNING_SOURCE_DIR}/rna/celltype/logs"
mkdir -p "${LOG_DIR}"

run_training_background "${LOG_DIR}/rna_celltype-train-gpt2-xl-$(date +%Y-%m-%d_%H-%M-%S).log" \
    molcrawl/models/gpt2/train.py \
    gpt2/configs/rna/train_gpt2_celltype_xl.py

echo "GPT-2 xl fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES})."
echo "Logs: ${LOG_DIR}/"
