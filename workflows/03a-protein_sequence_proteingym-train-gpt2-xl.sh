#!/bin/bash
# Fine-tune the protein_sequence GPT-2 (xl) model on ProteinGym DMS data.
#
# Prerequisites:
#   - protein_sequence GPT-2 xl pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/protein_sequence/gpt2-output/protein_sequence-xl/
#   - ProteinGym training_ready_hf_dataset must be prepared via
#       workflows/01-protein_sequence_proteingym-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03a-protein_sequence_proteingym-train-gpt2-xl.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir
NUM_GPUS=${NUM_GPUS:-1}
select_multi_gpu "$NUM_GPUS" 45

LOG_DIR="${LEARNING_SOURCE_DIR}/protein_sequence/proteingym/logs"
mkdir -p "${LOG_DIR}"

run_training_background "${LOG_DIR}/protein_sequence_proteingym-train-gpt2-xl-$(date +%Y-%m-%d_%H-%M-%S).log" \
    molcrawl/models/gpt2/train.py \
    molcrawl/tasks/pretrain/configs/protein_sequence/gpt2_proteingym_xl.py

echo "GPT-2 xl ProteinGym fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES})."
echo "Logs: ${LOG_DIR}/"
