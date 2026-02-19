#!/bin/bash
# RNAformer Medium Model Training Script
# RNA transcriptome learning with Geneformer-based architecture

# Set CUDA device (modify as needed)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Set learning source directory
export LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR:-learning_source_20250904-rna-refined}

# Weights & Biases configuration
export USE_WANDB=${USE_WANDB:-True}
export WANDB_PROJECT=${WANDB_PROJECT:-rnaformer-transcriptome}

# Training parameters
MODEL_SIZE="medium"
CONFIG_FILE="src/rnaformer/configs/rna.py"

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
python src/rnaformer/main.py \
    --config "${CONFIG_FILE}" \
    --model_size "${MODEL_SIZE}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "✅ Training completed! Log saved to: ${LOG_FILE}"
