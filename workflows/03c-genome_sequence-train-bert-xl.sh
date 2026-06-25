#!/bin/bash

set -e


# === XL-scale NVLink / NCCL optimisations (8-GPU AllReduce) ===
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Genome subset selector (matches the existing -small-subset wrapper convention).
GENOME_SUBSET="${GENOME_SUBSET:-mammal_centered}"
export GENOME_SUBSET

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/genome_sequence/logs
NUM_GPUS=${NUM_GPUS:-8}
select_multi_gpu "$NUM_GPUS" 40

LOG_FILE="${LEARNING_SOURCE_DIR}/genome_sequence/logs/genome_sequence-train-bert-xl-${GENOME_SUBSET}-$(date +%Y-%m-%d_%H-%M-%S).log"
run_training_background "$LOG_FILE" \
    molcrawl/models/bert/main.py \
    molcrawl/tasks/pretrain/configs/genome_sequence/bert_xl_subset.py