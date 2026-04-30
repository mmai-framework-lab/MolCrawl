#!/bin/bash
# Fine-tune the genome_sequence GPT-2 (medium) model on the ClinVar dataset.
#
# Prerequisites:
#   - genome_sequence GPT-2 medium pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/genome_sequence/gpt2-output/genome_sequence-medium/
#   - ClinVar training_ready_hf_dataset must be prepared via
#       workflows/01-genome_sequence_clinvar-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03a-genome_sequence_clinvar-train-gpt2-medium.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir
NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 20

mkdir -p ${LEARNING_SOURCE_DIR}/genome_sequence/clinvar/logs
LOG_FILE="${LEARNING_SOURCE_DIR}/genome_sequence/clinvar/logs/genome_sequence_clinvar-train-gpt2-medium-$(date +%Y-%m-%d_%H-%M-%S).log"
run_training_background "$LOG_FILE" \
    molcrawl/models/gpt2/train.py \
    ./molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_clinvar_medium.py

echo "GPT-2 medium ClinVar fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES})."
echo "Logs: ${LEARNING_SOURCE_DIR}/genome_sequence/clinvar/logs/"
