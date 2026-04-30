#!/bin/bash
#
# ESM-2 Training Script for Protein Sequence Data
#
# このスクリプトは既存のprotein_sequenceデータセット（UniProt）を使用して
# ESM-2モデルを学習します。
#
# 使用方法:
#   ./workflows/03e-protein_sequence-train-esm2-small.sh
#
# オプション:
#   CUDA_VISIBLE_DEVICES=0,1 で使用するGPUを指定
#   USE_WANDB=True で Weights & Biases によるログを有効化
#
# 例:
#   CUDA_VISIBLE_DEVICES=0 USE_WANDB=True ./workflows/03e-protein_sequence-train-esm2-small.sh
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
NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 10
# Set wandb settings from environment variables
USE_WANDB=${USE_WANDB:-False}
WANDB_PROJECT=${WANDB_PROJECT:-esm2-protein}
WANDB_ENTITY=${WANDB_ENTITY:-}

# Export environment variables for the training script
export USE_WANDB
export WANDB_PROJECT
export WANDB_ENTITY

# Log file path
LOG_FILE="${LEARNING_SOURCE_DIR}/protein_sequence/logs/esm2-train-small-$(date +%Y-%m-%d_%H-%M-%S).log"

echo "========================================"
echo "🧬 ESM-2 Training - Small Model"
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

# Run training in background
run_training_background "${LOG_FILE}" \
    molcrawl/models/esm2/main.py \
    molcrawl/models/esm2/configs/protein_sequence.py


