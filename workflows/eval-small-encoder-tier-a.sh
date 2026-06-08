#!/bin/bash
#SBATCH --job-name=small-encoder-tierA
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/small-encoder-tierA-%j.log
#SBATCH --error=slurm_logs/small-encoder-tierA-%j.log

# Tier A MLM perplexity / Top-1 accuracy for the 6 completed small encoder
# pretrains on the NEW (uncapped) corpus. Mirrors the layout of
# eval-medium-gpt2-tier-a.sh — same REPORT directory shape, same stdout
# markers — so the result can be grepped and added to the comparison
# report alongside the GPT-2 numbers.
#
# Targets each model's latest checkpoint-NNNN under bert-output / roberta-output.
# Eval is on each modality's own training_ready_hf_dataset valid split,
# capped at 2000 samples to keep total wall-clock under an hour.

set -e

# Host-specific paths are taken from the environment so this launcher stays
# portable across users / machines. Override any of these before submitting:
#   REPO_ROOT            repo checkout root (default: derived from this script's path)
#   LEARNING_SOURCE_DIR  REQUIRED — root holding the per-modality datasets/checkpoints
#   PYTHON               python interpreter to use (default: python on PATH)
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the learning source root before submitting}"
export LEARNING_SOURCE_DIR
LSD="${LEARNING_SOURCE_DIR}"
PYTHON="${PYTHON:-python}"

REPORT="${LSD}/protein_sequence/logs/small-encoder-tierA-${SLURM_JOB_ID}"
mkdir -p "$REPORT"

# Latest checkpoint helper — picks the highest-numbered checkpoint-NNNN/.
latest_ckpt() {
    ls -d "$1"/checkpoint-* 2>/dev/null | sort -V | tail -1
}

run() {  # label, ckpt_root, domain, arch, dataset_dir
    local label="$1" root="$2" domain="$3" arch="$4" ddir="$5"
    local ckpt
    ckpt=$(latest_ckpt "$root")
    if [ -z "$ckpt" ]; then echo "  [$label] SKIP — no checkpoint in $root"; return; fi
    local out="${REPORT}/${label}"; mkdir -p "$out"
    echo "  [$label] $(basename "$ckpt")"
    "${PYTHON}" scripts/eval_encoder_tier_a.py \
        --checkpoint_path "$ckpt" \
        --dataset_dir "$ddir" \
        --domain "$domain" \
        --arch "$arch" \
        --max_test_samples 2000 \
        --batch_size 8 \
        --device cuda \
        --output_dir "$out" > "${out}/stdout.log" 2>&1 \
        || echo "    eval errored — see ${out}/stdout.log"
    local ppl=$(grep -oE "perplexity: [0-9.]+" "${out}/stdout.log" | tail -1)
    local acc=$(grep -oE "Top-1 accuracy: [0-9.]+" "${out}/stdout.log" | tail -1)
    local loss=$(grep -oE "mlm_loss: [0-9.]+" "${out}/stdout.log" | tail -1)
    echo "    $loss | $ppl | $acc"
}

echo "==== small encoder Tier A job ${SLURM_JOB_ID} at $(date) ===="
echo
echo "### BERT small (3 modalities) ###"
run "protein-bert-small"     "${LSD}/protein_sequence/bert-output/protein_sequence-small"        protein_sequence    bert "${LSD}/protein_sequence/training_ready_hf_dataset"
run "genome-bert-small"      "${LSD}/genome_sequence/bert-output/genome_sequence-small"          genome_sequence     bert "${LSD}/genome_sequence/training_ready_hf_dataset"
run "molnatlang-bert-small"  "${LSD}/molecule_nat_lang/bert-output/molecule_nat_lang-small"      molecule_nat_lang   bert "${LSD}/molecule_nat_lang/training_ready_hf_dataset"
echo
echo "### RoBERTa small (3 modalities) ###"
run "protein-roberta-small"    "${LSD}/protein_sequence/roberta-output/protein_sequence-small"     protein_sequence    roberta "${LSD}/protein_sequence/training_ready_hf_dataset"
run "genome-roberta-small"     "${LSD}/genome_sequence/roberta-output/genome_sequence-small"       genome_sequence     roberta "${LSD}/genome_sequence/training_ready_hf_dataset"
run "molnatlang-roberta-small" "${LSD}/molecule_nat_lang/roberta-output/molecule_nat_lang-small"   molecule_nat_lang   roberta "${LSD}/molecule_nat_lang/training_ready_hf_dataset"
echo
echo "==== finished at $(date) ===="
echo "Report dir: ${REPORT}"
