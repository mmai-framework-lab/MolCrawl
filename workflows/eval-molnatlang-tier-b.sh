#!/bin/bash
#SBATCH --job-name=molnatlang-tier-b
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --output=../learning_source_20260520_uncapped/molecule_nat_lang/logs/tier-b-%j.out
#SBATCH --error=../learning_source_20260520_uncapped/molecule_nat_lang/logs/tier-b-%j.out

# Tier B verification for molecule_nat_lang × gpt2-small.
# See scripts/eval_molnatlang_tier_b.py.

set -e
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-../learning_source_20260520_uncapped}"
export LEARNING_SOURCE_DIR
PYTHON="${PYTHON:-python}"

# MoleculeNatLangTokenizer needs the GPT-2 tokenizer; compute nodes have no internet.
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-}"

OLD_CKPT="../learning_source_20260316/molecule_nat_lang/gpt2-output/molecule_nat_lang-small/ckpt.pt"
NEW_CKPT="../learning_source_20260520_uncapped/molecule_nat_lang/gpt2-output/molecule_nat_lang-small/ckpt.pt"
DATASET_DIR="../learning_source_20260520_uncapped/molecule_nat_lang/training_ready_hf_dataset"

OUT_JSON="${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs/tier-b-${SLURM_JOB_ID}.json"

echo "==== Tier B eval job ${SLURM_JOB_ID} starting at $(date) ===="
echo "Node: $(hostname)"

"${PYTHON}" scripts/eval_molnatlang_tier_b.py \
    --old_ckpt "${OLD_CKPT}" \
    --new_ckpt "${NEW_CKPT}" \
    --dataset_dir "${DATASET_DIR}" \
    --split valid \
    --max_samples 2000 \
    --top_k 20 \
    --out_json "${OUT_JSON}" \
    --device cuda

echo "==== finished at $(date) ===="
