#!/bin/bash
#SBATCH --job-name=molnatlang-prep-gpt2
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=4:00:00
#SBATCH --output=../learning_source_20260520_uncapped/molecule_nat_lang/logs/slurm-%j.out
#SBATCH --error=../learning_source_20260520_uncapped/molecule_nat_lang/logs/slurm-%j.out

# SLURM-side prepare_gpt2 for molecule_nat_lang number_sample=None uncapped-validation
# experiment. Peak RAM at concatenate_texts (batch_size=-1) is small (~3 GB on
# ~330M tokens), but we use the same 256 GB pattern as the protein run for
# consistency.

set -e
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-../learning_source_20260520_uncapped}"
export LEARNING_SOURCE_DIR
PYTHON="${PYTHON:-python}"

# Compute nodes have no outbound internet, and setup_cache_env points HF_HOME at
# the LSD-local empty cache. Point the molnatlang tokenizer loader at the GPT-2
# tokenizer snapshot already cached under $HOME (lustre, visible from every node).
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-}"

mkdir -p "${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs/molecule_nat_lang-prepare-gpt2-slurm-${TIMESTAMP}.log"

echo "==== SLURM job ${SLURM_JOB_ID} starting at $(date) ====" | tee "$LOG_FILE"
echo "Node: $(hostname), LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR}" | tee -a "$LOG_FILE"

"${PYTHON}" molcrawl/data/molecule_nat_lang/dataset/prepare_gpt2.py \
    "${LEARNING_SOURCE_DIR}/molecule_nat_lang.yaml" \
    2>&1 | tee -a "$LOG_FILE"

EXIT=${PIPESTATUS[0]}
echo "==== finished at $(date) (exit=${EXIT}) ====" | tee -a "$LOG_FILE"

if [ -d "${LEARNING_SOURCE_DIR}/molecule_nat_lang/training_ready_hf_dataset" ]; then
    du -sh "${LEARNING_SOURCE_DIR}/molecule_nat_lang/training_ready_hf_dataset" | tee -a "$LOG_FILE"
    "${PYTHON}" -c "
from datasets import load_from_disk
ds = load_from_disk('${LEARNING_SOURCE_DIR}/molecule_nat_lang/training_ready_hf_dataset')
for s in ds: print(f'{s}: {len(ds[s])} chunks of 1024 tokens')
" | tee -a "$LOG_FILE"
fi
exit ${EXIT}
