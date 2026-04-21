#!/bin/bash
# ChemBERTa-2 Small Model Training Script
# SMILES compounds learning with RoBERTa-based architecture

# Load common functions (sets $PYTHON)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Set learning source directory
export LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR:-learning_source_20251210}
check_learning_source_dir

# Multi-GPU support
NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 10

# Weights & Biases configuration
export USE_WANDB=${USE_WANDB:-True}
export WANDB_PROJECT=${WANDB_PROJECT:-chemberta2-compounds}

# Training parameters
MODEL_SIZE="small"
CONFIG_FILE="molcrawl/molcrawl/chemberta2/configs/compounds.py"

# Create log directory
LOG_DIR="${LEARNING_SOURCE_DIR}/compounds/logs"
mkdir -p "${LOG_DIR}"

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/chemberta2-train-${MODEL_SIZE}-${TIMESTAMP}.log"

echo "========================================"
echo "ChemBERTa-2 Training - ${MODEL_SIZE}"
echo "========================================"
echo "GPU:              ${CUDA_VISIBLE_DEVICES}"
echo "Num GPUs:         $(count_visible_gpus)"
echo "Learning source:  ${LEARNING_SOURCE_DIR}"
echo "Log file:         ${LOG_FILE}"
echo "========================================"
echo ""

# Run training (torchrun auto-detected for multi-GPU)
run_training molcrawl/chemberta2/main.py \
    --config "${CONFIG_FILE}" \
    --model_size "${MODEL_SIZE}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "Training completed! Log saved to: ${LOG_FILE}"
