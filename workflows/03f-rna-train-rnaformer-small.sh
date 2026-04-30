#!/bin/bash
# RNAformer Small Model Training Script
# RNA transcriptome learning with Geneformer-based architecture

# Load common functions (sets $PYTHON)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

# Set CUDA device (modify as needed)
NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 10
# Weights & Biases configuration
export USE_WANDB=${USE_WANDB:-True}
export WANDB_PROJECT=${WANDB_PROJECT:-rnaformer-transcriptome}

# Training parameters
MODEL_SIZE="small"
CONFIG_FILE="molcrawl/molcrawl/tasks/pretrain/configs/rna/rnaformer.py"

# Create log directory
LOG_DIR="${LEARNING_SOURCE_DIR}/rna/logs"
mkdir -p "${LOG_DIR}"

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/rnaformer-train-${MODEL_SIZE}-${TIMESTAMP}.log"

echo "🧬 Starting RNAformer ${MODEL_SIZE} training..."
echo "📊 GPU: ${CUDA_VISIBLE_DEVICES}"
echo "📁 Learning source: ${LEARNING_SOURCE_DIR}"
echo "📝 Log file: ${LOG_FILE}"
echo ""

# Run training
run_training molcrawl/models/rnaformer/main.py \
    --config "${CONFIG_FILE}" \
    --model_size "${MODEL_SIZE}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "✅ Training completed! Log saved to: ${LOG_FILE}"
