#!/bin/bash
#SBATCH --job-name=genome-eval-old-vs-new
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --output=../learning_source_20260520_uncapped/genome_sequence/logs/eval-old-vs-new-%j.out
#SBATCH --error=../learning_source_20260520_uncapped/genome_sequence/logs/eval-old-vs-new-%j.out

# Tier A cross-evaluation for genome_sequence × gpt2-small.
# OLD (50k cap = 2.0% of corpus, 379M tokens — uses single-file v1 parquet) vs
# NEW (500k cap = 20.2% of corpus, ~3 B tokens — same spm_tokenizer.model).
# Both evaluated on the same NEW valid split (held-out for OLD).

set -e
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-../learning_source_20260520_uncapped}"
export LEARNING_SOURCE_DIR
PYTHON="${PYTHON:-python}"

OLD_CKPT="../learning_source_20260316/genome_sequence/gpt2-output/genome_sequence-small/ckpt.pt"
NEW_CKPT="../learning_source_20260520_uncapped/genome_sequence/gpt2-output/genome_sequence-small/ckpt.pt"
SHARED_TEST_DIR="../learning_source_20260520_uncapped/genome_sequence/training_ready_hf_dataset"
TEST_PARAMS="{\"dataset_dir\": \"${SHARED_TEST_DIR}\"}"

# create_genome_tokenizer needs the spm_tokenizer.model path. Use the one copied
# alongside the NEW corpus (identical to OLD — was generated from full raw_files
# before the number_sample cap stage, so both runs trained against the same vocab).
GENOME_SPM_MODEL="../learning_source_20260520_uncapped/genome_sequence/spm_tokenizer.model"

REPORT_DIR="${LEARNING_SOURCE_DIR}/genome_sequence/logs/eval-old-vs-new-${SLURM_JOB_ID}"
mkdir -p "${REPORT_DIR}/old" "${REPORT_DIR}/new"

echo "==== genome eval-old-vs-new ${SLURM_JOB_ID} at $(date) ===="
echo "Node: $(hostname)"
echo "OLD: ${OLD_CKPT}"
echo "NEW: ${NEW_CKPT}"
echo "Tokenizer: ${GENOME_SPM_MODEL}"

echo "########## (1/2) OLD (50k cap, 379M tokens) ##########"
"${PYTHON}" molcrawl/models/gpt2/test_checkpoint.py \
    --checkpoint_path "${OLD_CKPT}" \
    --output_dir "${REPORT_DIR}/old" \
    --domain genome_sequence \
    --vocab_path "${GENOME_SPM_MODEL}" \
    --test_dataset_params "${TEST_PARAMS}" \
    --max_test_samples 2000 \
    --device cuda 2>&1 | tee "${REPORT_DIR}/old/stdout.log"

echo "########## (2/2) NEW (500k cap, ~3 B tokens) ##########"
"${PYTHON}" molcrawl/models/gpt2/test_checkpoint.py \
    --checkpoint_path "${NEW_CKPT}" \
    --output_dir "${REPORT_DIR}/new" \
    --domain genome_sequence \
    --vocab_path "${GENOME_SPM_MODEL}" \
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
