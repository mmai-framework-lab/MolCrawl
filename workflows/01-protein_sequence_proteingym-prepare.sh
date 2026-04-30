#!/bin/bash
# Download ProteinGym v1.3 DMS substitution data and prepare the
# training_ready_hf_dataset for protein_sequence fine-tuning.
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/01-protein_sequence_proteingym-prepare.sh
#
# Output:
#   $LEARNING_SOURCE_DIR/protein_sequence/proteingym/training_ready_hf_dataset/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

LOG_DIR="${LEARNING_SOURCE_DIR}/protein_sequence/proteingym/logs"
mkdir -p "${LOG_DIR}"

PYTHONUNBUFFERED=1 \
nohup bash -c '$PYTHON molcrawl/data/protein_sequence/preparation.py \
    assets/configs/protein_sequence.yaml --datasets proteingym' \
    > "${LOG_DIR}/protein_sequence_proteingym-prepare-$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

echo "ProteinGym data preparation running in background."
echo "Logs: ${LOG_DIR}/"
