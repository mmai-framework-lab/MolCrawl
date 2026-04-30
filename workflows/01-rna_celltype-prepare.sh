#!/bin/bash
# Download Geneformer cell type annotation dataset and prepare it for
# GPT-2 / BERT fine-tuning.
#
# Steps performed:
#   1. Download cell_type_train_data.dataset from ctheodoris/Genecorpus-30M
#      on HuggingFace Hub.
#   2. Run prepare_celltype.py to split (80/10/10), chunk to context_length
#      1024, and save as training_ready_hf_dataset format.
#
# Prerequisites:
#   - LEARNING_SOURCE_DIR must be set.
#   - (Optional) HuggingFace token for private repos: huggingface-cli login
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/01-rna_celltype-prepare.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

LOG_DIR="${LEARNING_SOURCE_DIR}/rna/celltype/logs"
mkdir -p "${LOG_DIR}"

echo "[1/1] Preparing RNA CellType dataset (download + split + chunk)..."
nohup $PYTHON molcrawl/data/rna/preparation.py \
    assets/configs/rna.yaml \
    --datasets celltype \
    > "${LOG_DIR}/rna_celltype-prepare-$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

echo "Preparation running in background. Logs: ${LOG_DIR}/"
