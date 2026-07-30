#!/bin/bash
#SBATCH --job-name=molnatlang-tierA-sizes
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=2:00:00
#SBATCH --output=../learning_source_20260520_uncapped/molecule_nat_lang/logs/tierA-sizes-%j.out
#SBATCH --error=../learning_source_20260520_uncapped/molecule_nat_lang/logs/tierA-sizes-%j.out

# Tier A perplexity for the completed molnatlang medium/large/xl runs, on the
# shared NEW valid split. gpt2 has OLD (50k-cap) baselines in
# learning_source_20260316; llama has none (llama was 0-ckpt pre-fix), so its
# rows are NEW-only.

set -e
REPO_ROOT="/lustre/home/matsubara/riken-dataset-fundational-model"
cd "${REPO_ROOT}"

LEARNING_SOURCE_DIR="../learning_source_20260520_uncapped"
export LEARNING_SOURCE_DIR
PYTHON="/lustre/home/matsubara/miniforge3/envs/molcrawl/bin/python"
export GPT2_TOKENIZER_DIR="/lustre/home/matsubara/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

NEW_LSD="../learning_source_20260520_uncapped"
OLD_LSD="../learning_source_20260316"
TEST_DIR="${NEW_LSD}/molecule_nat_lang/training_ready_hf_dataset"
TEST_PARAMS="{\"dataset_dir\": \"${TEST_DIR}\"}"
REPORT_DIR="${NEW_LSD}/molecule_nat_lang/logs/tierA-sizes-${SLURM_JOB_ID}"
mkdir -p "$REPORT_DIR"

eval_one() {
    local label="$1" ckpt="$2"
    [ -f "$ckpt" ] || { echo "  [$label] SKIP (no ckpt: $ckpt)"; return; }
    local out="${REPORT_DIR}/${label}"
    mkdir -p "$out"
    "${PYTHON}" molcrawl/models/gpt2/test_checkpoint.py \
        --checkpoint_path "$ckpt" \
        --output_dir "$out" \
        --domain molecule_nat_lang \
        --test_dataset_params "${TEST_PARAMS}" \
        --max_test_samples 2000 \
        --device cuda > "${out}/stdout.log" 2>&1 || echo "  [$label] eval errored"
    local ppl=$(tr '\r' '\n' < "${out}/stdout.log" | grep -oE "perplexity: [0-9.]+" | tail -1)
    local acc=$(tr '\r' '\n' < "${out}/stdout.log" | grep -oE "Top-1 accuracy: [0-9.]+" | tail -1)
    echo "  [$label] $ppl | $acc"
}

echo "==== molnatlang Tier A (sizes) job ${SLURM_JOB_ID} at $(date) ===="
echo
echo "### gpt2 large (OLD vs NEW) ###"
eval_one "gpt2-large-OLD" "${OLD_LSD}/molecule_nat_lang/gpt2-output/molecule_nat_lang-large/ckpt.pt"
eval_one "gpt2-large-NEW" "${NEW_LSD}/molecule_nat_lang/gpt2-output/molecule_nat_lang-large/ckpt.pt"
echo
echo "### gpt2 xl (OLD vs NEW) ###"
eval_one "gpt2-xl-OLD" "${OLD_LSD}/molecule_nat_lang/gpt2-output/molecule_nat_lang-ex-large/ckpt.pt"
eval_one "gpt2-xl-NEW" "${NEW_LSD}/molecule_nat_lang/gpt2-output/molecule_nat_lang-ex-large/ckpt.pt"
echo
echo "### llama medium/large/xl (NEW only — no OLD baseline) ###"
eval_one "llama-medium-NEW" "${NEW_LSD}/molecule_nat_lang/llama-output/molecule_nat_lang-medium/ckpt.pt"
eval_one "llama-large-NEW"  "${NEW_LSD}/molecule_nat_lang/llama-output/molecule_nat_lang-large/ckpt.pt"
eval_one "llama-xl-NEW"     "${NEW_LSD}/molecule_nat_lang/llama-output/molecule_nat_lang-ex-large/ckpt.pt"
echo
echo "==== finished at $(date) ===="
