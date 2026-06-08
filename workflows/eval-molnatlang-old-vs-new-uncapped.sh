#!/bin/bash
#SBATCH --job-name=molnatlang-eval-old-vs-new
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --output=../learning_source_20260520_uncapped/molecule_nat_lang/logs/eval-old-vs-new-%j.out
#SBATCH --error=../learning_source_20260520_uncapped/molecule_nat_lang/logs/eval-old-vs-new-%j.out

# Tier A cross-evaluation for molecule_nat_lang × gpt2-small.
# OLD (50k cap, 4.2M tokens) vs NEW (full ~3.3M-row corpus, ~330M tokens).
# Both evaluated on the same NEW valid split (effectively held-out for OLD).

set -e
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-../learning_source_20260520_uncapped}"
export LEARNING_SOURCE_DIR
PYTHON="${PYTHON:-python}"

# MoleculeNatLangTokenizer needs GPT-2 tokenizer; compute nodes have no internet.
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-}"

OLD_CKPT="../learning_source_20260316/molecule_nat_lang/gpt2-output/molecule_nat_lang-small/ckpt.pt"
NEW_CKPT="../learning_source_20260520_uncapped/molecule_nat_lang/gpt2-output/molecule_nat_lang-small/ckpt.pt"
SHARED_TEST_DIR="../learning_source_20260520_uncapped/molecule_nat_lang/training_ready_hf_dataset"
TEST_PARAMS="{\"dataset_dir\": \"${SHARED_TEST_DIR}\"}"

REPORT_DIR="${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs/eval-old-vs-new-${SLURM_JOB_ID}"
mkdir -p "${REPORT_DIR}/old" "${REPORT_DIR}/new"

echo "==== molnatlang eval-old-vs-new ${SLURM_JOB_ID} at $(date) ===="
echo "Node: $(hostname)"
echo "OLD: ${OLD_CKPT}"
echo "NEW: ${NEW_CKPT}"

echo "########## (1/2) OLD (50k cap, ~4.2M tokens) ##########"
"${PYTHON}" molcrawl/models/gpt2/test_checkpoint.py \
    --checkpoint_path "${OLD_CKPT}" \
    --output_dir "${REPORT_DIR}/old" \
    --domain molecule_nat_lang \
    --test_dataset_params "${TEST_PARAMS}" \
    --max_test_samples 2000 \
    --device cuda 2>&1 | tee "${REPORT_DIR}/old/stdout.log"

echo "########## (2/2) NEW (full corpus, ~330M tokens) ##########"
"${PYTHON}" molcrawl/models/gpt2/test_checkpoint.py \
    --checkpoint_path "${NEW_CKPT}" \
    --output_dir "${REPORT_DIR}/new" \
    --domain molecule_nat_lang \
    --test_dataset_params "${TEST_PARAMS}" \
    --max_test_samples 2000 \
    --device cuda 2>&1 | tee "${REPORT_DIR}/new/stdout.log"

echo
echo "########## summary ##########"
for tag in old new; do
  echo "== $tag =="
  tr '\r' '\n' < "${REPORT_DIR}/${tag}/stdout.log" | grep -iE "✓ perplexity:|Top-1 accuracy:|Top-5 accuracy:" | head -3
done
echo "==== finished at $(date) ===="
