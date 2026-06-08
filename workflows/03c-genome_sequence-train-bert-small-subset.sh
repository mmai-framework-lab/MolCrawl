#!/bin/bash
# Pretrain genome BERT (small) on a single Evo2-derived subset.
#
# Usage:
#   export LEARNING_SOURCE_DIR=/path/to/learning_source
#   conda activate molcrawl
#   GENOME_SUBSET=mammal_centered bash workflows/03c-genome_sequence-train-bert-small-subset.sh
#
# Or via SLURM:
#   sbatch --gres=gpu:1 --export=ALL,GENOME_SUBSET=mammal_centered \
#       workflows/03c-genome_sequence-train-bert-small-subset.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
LOG_FILE="${LOG_DIR}/${GENOME_SUBSET}-bert-small-$(date +%Y-%m-%d_%H-%M-%S).log"

export GENOME_SUBSET   # required by bert_small_subset.py
run_training_background "$LOG_FILE" \
    molcrawl/models/bert/main.py \
    molcrawl/tasks/pretrain/configs/genome_sequence/bert_small_subset.py
