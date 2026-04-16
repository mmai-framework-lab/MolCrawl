#!/bin/bash
#
# DNABERT-2 Training Script for Genome Sequence Data (Large Model)
#

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

# Create log directory
mkdir -p ${LEARNING_SOURCE_DIR}/genome_sequence/logs

# Set default GPU if not specified (large model may need multiple GPUs)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Set wandb settings from environment variables
USE_WANDB=${USE_WANDB:-False}
WANDB_PROJECT=${WANDB_PROJECT:-dnabert2-genome}
WANDB_ENTITY=${WANDB_ENTITY:-}

# Export environment variables
export USE_WANDB
export WANDB_PROJECT
export WANDB_ENTITY

# Log file path
LOG_FILE="${LEARNING_SOURCE_DIR}/genome_sequence/logs/dnabert2-train-large-$(date +%Y-%m-%d_%H-%M-%S).log"

echo "========================================"
echo "🧬 DNABERT-2 Training - Large Model"
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
echo "⚠️  Warning: Large model requires significant GPU memory"
echo "    Recommended: 4x A100 40GB or equivalent"
echo ""
echo "Starting training in background..."
echo "Monitor progress with:"
echo "  tail -f ${LOG_FILE}"
echo ""

# Run training in background with large model size override
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nohup bash -c \
    '$PYTHON molcrawl/dnabert2/main.py molcrawl/dnabert2/configs/genome_sequence.py --model_size=large' \
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
