#!/bin/bash
#SBATCH --job-name=protein-tier-b
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --output=../learning_source_20260520_uncapped/protein_sequence/logs/tier-b-%j.out
#SBATCH --error=../learning_source_20260520_uncapped/protein_sequence/logs/tier-b-%j.out

# Tier B verification — per-token + per-position cross-entropy comparison.
# See scripts/eval_protein_tier_b.py and docs/_tmp/20260520-number-sample-impact-retrain-list.md §9.3.

set -e
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-../learning_source_20260520_uncapped}"
export LEARNING_SOURCE_DIR
PYTHON="${PYTHON:-python}"

OLD_CKPT="../learning_source_20260316/protein_sequence/gpt2-output/protein_sequence-small/ckpt.pt"
NEW_CKPT="../learning_source_20260520_uncapped/protein_sequence/gpt2-output/protein_sequence-small/ckpt.pt"
DATASET_DIR="../learning_source_20260520_uncapped/protein_sequence/training_ready_hf_dataset"

OUT_JSON="${LEARNING_SOURCE_DIR}/protein_sequence/logs/tier-b-${SLURM_JOB_ID}.json"

echo "==== Tier B eval job ${SLURM_JOB_ID} starting at $(date) ===="
echo "Node: $(hostname)"

"${PYTHON}" scripts/eval_protein_tier_b.py \
    --old_ckpt "${OLD_CKPT}" \
    --new_ckpt "${NEW_CKPT}" \
    --dataset_dir "${DATASET_DIR}" \
    --split valid \
    --max_samples 2000 \
    --out_json "${OUT_JSON}" \
    --device cuda

echo "==== finished at $(date) ===="
