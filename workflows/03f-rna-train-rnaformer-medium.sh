#!/bin/bash
# RNAformer Medium Model Training Script
# RNA transcriptome learning with Geneformer-based architecture

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

NUM_GPUS=${NUM_GPUS:-1}

select_multi_gpu "$NUM_GPUS" 20
export USE_WANDB=${USE_WANDB:-False}
export WANDB_PROJECT=${WANDB_PROJECT:-rnaformer-transcriptome}

MODEL_SIZE="medium"

LOG_DIR="${LEARNING_SOURCE_DIR}/rna/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/rnaformer-train-${MODEL_SIZE}-$(date +%Y-%m-%d_%H-%M-%S).log"

run_training_background "${LOG_FILE}" \
    molcrawl/models/rnaformer/main.py \
    --config molcrawl/models/rnaformer/configs/rna.py --model_size ${MODEL_SIZE}
