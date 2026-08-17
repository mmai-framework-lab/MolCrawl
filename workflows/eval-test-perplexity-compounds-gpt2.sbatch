#!/bin/bash
#SBATCH --job-name=mc-cmp-testppl
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Held-out perplexity for the compounds GPT-2 ladder (the other half of the
# evaluation the compounds plan asks for, alongside MoleculeNet).
#
# The ladder selected its checkpoints on valid, so test has stayed held out and is
# reportable. Scoring walks the packed test blocks with the same causal objective
# training used, which keeps the number comparable to the val losses in the logs.
#
# Submit from the repository root:
#   LEARNING_SOURCE_DIR=<packed dir> sbatch workflows/eval-test-perplexity-compounds-gpt2.sbatch
set -e
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR before submitting}"
SIZES="${SIZES:-small medium large ex-large}"
SPLIT="${SPLIT:-test}"
MAX_BLOCKS="${MAX_BLOCKS:-0}"   # 0 = whole split
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
OUT="${OUT_DIR:-${LEARNING_SOURCE_DIR}/experiment_data/eval/test-perplexity}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p "${OUT}" workflows/slurm-logs

echo "=== compounds GPT-2 ${SPLIT} perplexity ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "sizes: ${SIZES} | split=${SPLIT} | max_blocks=${MAX_BLOCKS:-all}"
echo "code : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
"${REPO}/miniconda/bin/python" scripts/eval_gpt2_perplexity.py \
  --sizes ${SIZES} --split "${SPLIT}" --max-blocks "${MAX_BLOCKS}" --output-dir "${OUT}"
echo "=== rc=$? @ $(date -Is) ==="
