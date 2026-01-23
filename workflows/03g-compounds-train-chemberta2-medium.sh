#!/bin/bash
# ChemBERTa-2 Medium Model Training Script
# SMILES compounds learning with RoBERTa-based architecture

# Set CUDA device (modify as needed)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Set learning source directory
export LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR:-learning_source_20251210}

# Weights & Biases configuration
export USE_WANDB=${USE_WANDB:-True}
export WANDB_PROJECT=${WANDB_PROJECT:-chemberta2-compounds}

# Training parameters
MODEL_SIZE="medium"
CONFIG_FILE="chemberta2/configs/compounds.py"

# Create log directory
LOG_DIR="${LEARNING_SOURCE_DIR}/compounds/logs"
mkdir -p "${LOG_DIR}"

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/chemberta2-train-${MODEL_SIZE}-${TIMESTAMP}.log"

echo "🧪 Starting ChemBERTa-2 ${MODEL_SIZE} training..."
echo "📊 GPU: ${CUDA_VISIBLE_DEVICES}"
echo "📁 Learning source: ${LEARNING_SOURCE_DIR}"
echo "📝 Log file: ${LOG_FILE}"
echo ""

# Run training
python chemberta2/main.py \
    --config "${CONFIG_FILE}" \
    --model_size "${MODEL_SIZE}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "✅ Training completed! Log saved to: ${LOG_FILE}"
