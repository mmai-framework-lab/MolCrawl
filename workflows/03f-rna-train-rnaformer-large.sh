#!/bin/bash
# RNAformer Large Model Training Script
# RNA transcriptome learning with Geneformer-based architecture

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export USE_WANDB=${USE_WANDB:-False}
export WANDB_PROJECT=${WANDB_PROJECT:-rnaformer-transcriptome}

MODEL_SIZE="large"

LOG_DIR="${LEARNING_SOURCE_DIR}/rna/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/rnaformer-train-${MODEL_SIZE}-$(date +%Y-%m-%d_%H-%M-%S).log"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nohup bash -c \
    '$PYTHON molcrawl/rnaformer/main.py --config molcrawl/rnaformer/configs/rna.py --model_size '"${MODEL_SIZE}" \
    > "${LOG_FILE}" 2>&1 &
