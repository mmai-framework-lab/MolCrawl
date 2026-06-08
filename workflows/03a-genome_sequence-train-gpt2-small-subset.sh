#!/bin/bash
# Pretrain genome GPT-2 (small) on a single Evo2-derived subset.
#
# Usage:
#   export LEARNING_SOURCE_DIR=/path/to/learning_source
#   conda activate molcrawl
#   GENOME_SUBSET=mammal_centered bash workflows/03a-genome_sequence-train-gpt2-small-subset.sh
#
# Or via SLURM:
#   sbatch --gres=gpu:1 --export=ALL,GENOME_SUBSET=mammal_centered \
#       workflows/03a-genome_sequence-train-gpt2-small-subset.sh
set -e

# Resolve workflows/ dir. Under sbatch, ${BASH_SOURCE[0]} points at the SLURM
# spool copy, so prefer SLURM_SUBMIT_DIR (the cwd at submit time) when set.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/workflows/common_functions.sh" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/workflows"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

if [ -z "${GENOME_SUBSET:-}" ]; then
    echo "ERROR: GENOME_SUBSET env var is required."
    echo "Example: GENOME_SUBSET=mammal_centered bash $0"
    exit 1
fi

NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 10

LOG_DIR="${LEARNING_SOURCE_DIR}/genome_sequence/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${GENOME_SUBSET}-gpt2-small-$(date +%Y-%m-%d_%H-%M-%S).log"

export GENOME_SUBSET   # required by gpt2_small_subset.py
run_training_background "$LOG_FILE" \
    molcrawl/models/gpt2/train.py \
    molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_subset.py
