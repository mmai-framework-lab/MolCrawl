#!/bin/bash
#
# ESM-2 Training Script for Protein Sequence Data (Medium Model)
#

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

# Create log directory
mkdir -p ${LEARNING_SOURCE_DIR}/protein_sequence/logs

# Set default GPU if not specified
NUM_GPUS=${NUM_GPUS:-2}
select_multi_gpu "$NUM_GPUS" 20
# Set wandb settings from environment variables
USE_WANDB=${USE_WANDB:-False}
WANDB_PROJECT=${WANDB_PROJECT:-esm2-protein}
WANDB_ENTITY=${WANDB_ENTITY:-}

# Export environment variables
export USE_WANDB
export WANDB_PROJECT
export WANDB_ENTITY

# Log file path
LOG_FILE="${LEARNING_SOURCE_DIR}/protein_sequence/logs/esm2-train-medium-$(date +%Y-%m-%d_%H-%M-%S).log"

echo "========================================"
echo "🧬 ESM-2 Training - Medium Model"
echo "========================================"
echo "GPU:              ${CUDA_VISIBLE_DEVICES}"
echo "Log file:         ${LOG_FILE}"
echo "Wandb:            ${USE_WANDB}"
if [ "$USE_WANDB" = "True" ]; then
    echo "Wandb project:    ${WANDB_PROJECT}"
    echo "Wandb entity:     ${WANDB_ENTITY}"
fi
echo "========================================"
echo ""
echo "Starting training in background..."
echo "Monitor progress with:"
echo "  tail -f ${LOG_FILE}"
echo ""

# Run training in background with medium model size override
run_training_background "${LOG_FILE}" \
    molcrawl/models/esm2/main.py \
    molcrawl/tasks/pretrain/configs/protein_sequence/esm2.py --model_size=medium


