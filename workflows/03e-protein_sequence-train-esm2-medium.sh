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
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

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
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nohup bash -c \
    'python src/esm2/main.py esm2/configs/protein_sequence.py --model_size=medium' \
    > "${LOG_FILE}" 2>&1 &

# Get PID
PID=$!

echo "✅ Training started (PID: ${PID})"
echo ""
echo "Useful commands:"
echo "  # Monitor log:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "  # Check if training is running:"
echo "  ps aux | grep ${PID}"
echo ""
echo "  # Stop training:"
echo "  kill ${PID}"
echo ""
