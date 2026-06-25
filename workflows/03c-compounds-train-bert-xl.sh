#!/bin/bash

set -e


# === XL-scale NVLink / NCCL optimisations (8-GPU AllReduce) ===
# Boss spec assumes B200; H200 also benefits. On AMD MI300X (gpu04)
# these are inert (no NVLink concept). Adjust NCCL_IB_DISABLE if
# running multi-node with InfiniBand.
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/compounds/logs
NUM_GPUS=${NUM_GPUS:-8}
select_multi_gpu "$NUM_GPUS" 40

LOG_FILE="${LEARNING_SOURCE_DIR}/compounds/logs/compounds-train-bert-xl-$(date +%Y-%m-%d_%H-%M-%S).log"
run_training_background "$LOG_FILE" \
    molcrawl/models/bert/main.py \
    molcrawl/tasks/pretrain/configs/compounds/bert_xl.py