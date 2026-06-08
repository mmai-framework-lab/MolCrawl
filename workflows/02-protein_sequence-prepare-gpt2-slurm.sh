#!/bin/bash
#SBATCH --job-name=protein-prep-gpt2
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=../learning_source_20260520_uncapped/protein_sequence/logs/slurm-%j.out
#SBATCH --error=../learning_source_20260520_uncapped/protein_sequence/logs/slurm-%j.out

# SLURM-side rerun of prepare_gpt2 for the protein_sequence number_sample=5M
# uncapped-validation experiment. The login node's 61 GiB RAM kept getting
# OOM-killed at concatenate_texts (batch_size=-1) — the materialised int64
# list for 4M tokenized train rows alone peaks ~14 GiB, plus the source
# DatasetDict stays mapped, plus other users on the shared box. 256 GiB on
# the h200-long node gives the concatenate step ~4x headroom.
#
# GPU allocation is mandatory on h200-long (GRES requirement) but unused —
# prepare_gpt2.py is pure CPU+RAM.
#
# Input artefacts already staged under ../learning_source_20260520_uncapped/:
#   - protein_sequence/raw_files/      (copied from learning_source_20260316)
#   - protein_sequence/*.marker        (resume markers)
#   - protein_sequence.yaml            (override YAML with number_sample=5_000_000)
#   - .cache/huggingface/datasets/     (partial tokenize cache from earlier
#                                       killed runs; HF will reuse on fingerprint match)

set -e

# sbatch copies this script to /var/spool/slurmd/jobN/, so BASH_SOURCE resolves
# there. Anchor paths to the project root explicitly.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-../learning_source_20260520_uncapped}"
export LEARNING_SOURCE_DIR
PYTHON="${PYTHON:-python}"

mkdir -p "${LEARNING_SOURCE_DIR}/protein_sequence/logs"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="${LEARNING_SOURCE_DIR}/protein_sequence/logs/protein_sequence-prepare-gpt2-slurm-${TIMESTAMP}.log"

echo "==== SLURM job ${SLURM_JOB_ID} starting at $(date) ====" | tee "$LOG_FILE"
echo "Node: $(hostname)" | tee -a "$LOG_FILE"
echo "LEARNING_SOURCE_DIR: ${LEARNING_SOURCE_DIR}" | tee -a "$LOG_FILE"
echo "Memory limit: ${SLURM_MEM_PER_NODE} MB" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

"${PYTHON}" molcrawl/data/protein_sequence/dataset/prepare_gpt2.py \
    "${LEARNING_SOURCE_DIR}/protein_sequence.yaml" \
    2>&1 | tee -a "$LOG_FILE"

EXIT=${PIPESTATUS[0]}
echo "" | tee -a "$LOG_FILE"
echo "==== SLURM job ${SLURM_JOB_ID} finished at $(date) (exit=${EXIT}) ====" | tee -a "$LOG_FILE"

if [ -d "${LEARNING_SOURCE_DIR}/protein_sequence/training_ready_hf_dataset" ]; then
    echo "training_ready_hf_dataset created:" | tee -a "$LOG_FILE"
    du -sh "${LEARNING_SOURCE_DIR}/protein_sequence/training_ready_hf_dataset" | tee -a "$LOG_FILE"
    "${PYTHON}" -c "
from datasets import load_from_disk
ds = load_from_disk('${LEARNING_SOURCE_DIR}/protein_sequence/training_ready_hf_dataset')
for s in ds: print(f'{s}: {len(ds[s])} chunks of 1024 tokens')
" | tee -a "$LOG_FILE"
fi

exit ${EXIT}
