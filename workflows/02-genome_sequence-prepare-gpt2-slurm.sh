#!/bin/bash
#SBATCH --job-name=genome-prep-gpt2
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=../learning_source_20260520_uncapped/genome_sequence/logs/slurm-%j.out
#SBATCH --error=../learning_source_20260520_uncapped/genome_sequence/logs/slurm-%j.out

# SLURM-side prepare_gpt2 for genome_sequence number_sample=None uncapped-validation
# experiment. The single 27 GB v1 .parquet file (copied from learning_source_20260316
# via rsync, since the v1 format predates the int32-offset shard fix) is read directly.
# Peak RAM at concatenate_texts (batch_size=-1) is ~121 GB (train split's 15B tokens
# materialised as int64 list). Fits on the 256 GB node with ~50% headroom.

set -e
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-../learning_source_20260520_uncapped}"
export LEARNING_SOURCE_DIR
PYTHON="${PYTHON:-python}"

mkdir -p "${LEARNING_SOURCE_DIR}/genome_sequence/logs"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="${LEARNING_SOURCE_DIR}/genome_sequence/logs/genome-prepare-gpt2-slurm-${TIMESTAMP}.log"

echo "==== SLURM job ${SLURM_JOB_ID} starting at $(date) ====" | tee "$LOG_FILE"
echo "Node: $(hostname), LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR}" | tee -a "$LOG_FILE"

"${PYTHON}" molcrawl/data/genome_sequence/dataset/prepare_gpt2.py \
    "${LEARNING_SOURCE_DIR}/genome_sequence.yaml" \
    2>&1 | tee -a "$LOG_FILE"

EXIT=${PIPESTATUS[0]}
echo "==== finished at $(date) (exit=${EXIT}) ====" | tee -a "$LOG_FILE"

if [ -d "${LEARNING_SOURCE_DIR}/genome_sequence/training_ready_hf_dataset" ]; then
    du -sh "${LEARNING_SOURCE_DIR}/genome_sequence/training_ready_hf_dataset" | tee -a "$LOG_FILE"
    "${PYTHON}" -c "
from datasets import load_from_disk
ds = load_from_disk('${LEARNING_SOURCE_DIR}/genome_sequence/training_ready_hf_dataset')
for s in ds: print(f'{s}: {len(ds[s])} chunks of 1024 tokens')
" | tee -a "$LOG_FILE"
fi
exit ${EXIT}
