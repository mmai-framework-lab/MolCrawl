#!/bin/bash
#SBATCH --job-name=genome-tier-b
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --output=../learning_source_20260520_uncapped/genome_sequence/logs/tier-b-%j.out
#SBATCH --error=../learning_source_20260520_uncapped/genome_sequence/logs/tier-b-%j.out

# Tier B verification for genome_sequence × gpt2-small.
# Diagnoses *where* (per BPE piece, per chunk-position) the OLD (50k cap, 89k iter)
# and NEW (500k cap, 50k iter — undertrained per Tier A 18213) differ. Feeds the
# decision to extend NEW training (see workflows/03a-genome_sequence-train-small-extend.sh
# + molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_extend.py).

set -e
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-../learning_source_20260520_uncapped}"
export LEARNING_SOURCE_DIR
PYTHON="${PYTHON:-python}"

OLD_CKPT="../learning_source_20260316/genome_sequence/gpt2-output/genome_sequence-small/ckpt.pt"
NEW_CKPT="../learning_source_20260520_uncapped/genome_sequence/gpt2-output/genome_sequence-small/ckpt.pt"
DATASET_DIR="../learning_source_20260520_uncapped/genome_sequence/training_ready_hf_dataset"
SPM_MODEL="../learning_source_20260520_uncapped/genome_sequence/spm_tokenizer.model"

OUT_JSON="${LEARNING_SOURCE_DIR}/genome_sequence/logs/tier-b-${SLURM_JOB_ID}.json"

echo "==== Tier B eval job ${SLURM_JOB_ID} starting at $(date) ===="
echo "Node: $(hostname)"

"${PYTHON}" scripts/eval_genome_tier_b.py \
    --old_ckpt "${OLD_CKPT}" \
    --new_ckpt "${NEW_CKPT}" \
    --dataset_dir "${DATASET_DIR}" \
    --spm_model "${SPM_MODEL}" \
    --split valid \
    --max_samples 2000 \
    --top_k 20 \
    --out_json "${OUT_JSON}" \
    --device cuda

echo "==== finished at $(date) ===="
