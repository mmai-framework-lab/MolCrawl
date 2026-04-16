#!/bin/bash
# ChemBERTa-2 Large Model Training Script
# SMILES compounds learning with RoBERTa-based architecture

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export USE_WANDB=${USE_WANDB:-False}
export WANDB_PROJECT=${WANDB_PROJECT:-chemberta2-compounds}

MODEL_SIZE="large"

LOG_DIR="${LEARNING_SOURCE_DIR}/compounds/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/chemberta2-train-${MODEL_SIZE}-$(date +%Y-%m-%d_%H-%M-%S).log"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nohup bash -c \
    '$PYTHON molcrawl/chemberta2/main.py --config molcrawl/chemberta2/configs/compounds.py --model_size '"${MODEL_SIZE}" \
    > "${LOG_FILE}" 2>&1 &
