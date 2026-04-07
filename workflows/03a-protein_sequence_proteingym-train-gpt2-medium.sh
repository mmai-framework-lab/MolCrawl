#!/bin/bash
# Fine-tune the protein_sequence GPT-2 (medium) model on ProteinGym DMS data.
#
# Prerequisites:
#   - protein_sequence GPT-2 medium pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/protein_sequence/gpt2-output/protein_sequence-medium/
#   - ProteinGym training_ready_hf_dataset must be prepared via
#       workflows/01-protein_sequence_proteingym-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03a-protein_sequence_proteingym-train-gpt2-medium.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir
auto_select_gpu 20

LOG_DIR="${LEARNING_SOURCE_DIR}/protein_sequence/proteingym/logs"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} PYTHONUNBUFFERED=1 \
nohup bash -c '$PYTHON molcrawl/gpt2/train.py \
    gpt2/configs/protein_sequence/train_gpt2_proteingym_medium.py' \
    > "${LOG_DIR}/protein_sequence_proteingym-train-gpt2-medium-$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

echo "GPT-2 medium ProteinGym fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES})."
echo "Logs: ${LOG_DIR}/"
