#!/bin/bash
#SBATCH --job-name=medium-gpt2-tierA
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=2:00:00
#SBATCH --output=../learning_source_20260520_uncapped/protein_sequence/logs/medium-tierA-%j.out
#SBATCH --error=../learning_source_20260520_uncapped/protein_sequence/logs/medium-tierA-%j.out

# Tier A perplexity for the completed gpt2-medium runs (protein/molnatlang/genome),
# OLD (50k cap, learning_source_20260316) vs NEW (uncapped). Each evaluated on the
# NEW modality's own valid split (held-out for OLD).

set -e
REPO_ROOT="/lustre/home/matsubara/riken-dataset-fundational-model"
cd "${REPO_ROOT}"
LEARNING_SOURCE_DIR="../learning_source_20260520_uncapped"
export LEARNING_SOURCE_DIR
PYTHON="/lustre/home/matsubara/miniforge3/envs/molcrawl/bin/python"
export GPT2_TOKENIZER_DIR="/lustre/home/matsubara/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

NEW="../learning_source_20260520_uncapped"
OLD="../learning_source_20260316"
REPORT="${NEW}/protein_sequence/logs/medium-tierA-${SLURM_JOB_ID}"
mkdir -p "$REPORT"

GENOME_SPM="${NEW}/genome_sequence/spm_tokenizer.model"

run() {  # label, ckpt, domain, dataset_dir, [vocab_path]
    local label="$1" ckpt="$2" domain="$3" ddir="$4" vocab="$5"
    [ -f "$ckpt" ] || { echo "  [$label] SKIP no ckpt"; return; }
    local out="${REPORT}/${label}"; mkdir -p "$out"
    local vp=(); [ -n "$vocab" ] && vp=(--vocab_path "$vocab")
    "${PYTHON}" molcrawl/models/gpt2/test_checkpoint.py \
        --checkpoint_path "$ckpt" --output_dir "$out" \
        --domain "$domain" --test_dataset_params "{\"dataset_dir\": \"${ddir}\"}" \
        --max_test_samples 2000 --device cuda "${vp[@]}" > "${out}/stdout.log" 2>&1 \
        || echo "  [$label] eval errored (see ${out}/stdout.log)"
    local ppl=$(tr '\r' '\n' < "${out}/stdout.log" | grep -oE "perplexity: [0-9.]+" | tail -1)
    local acc=$(tr '\r' '\n' < "${out}/stdout.log" | grep -oE "Top-1 accuracy: [0-9.]+" | tail -1)
    echo "  [$label] $ppl | $acc"
}

echo "==== medium gpt2 Tier A job ${SLURM_JOB_ID} at $(date) ===="
echo
echo "### protein medium (OLD vs NEW) ###"
run "protein-medium-OLD" "${OLD}/protein_sequence/gpt2-output/protein_sequence-medium/ckpt.pt" protein_sequence "${NEW}/protein_sequence/training_ready_hf_dataset"
run "protein-medium-NEW" "${NEW}/protein_sequence/gpt2-output/protein_sequence-medium/ckpt.pt" protein_sequence "${NEW}/protein_sequence/training_ready_hf_dataset"
echo
echo "### molnatlang medium (OLD vs NEW) ###"
run "molnatlang-medium-OLD" "${OLD}/molecule_nat_lang/gpt2-output/molecule_nat_lang-medium/ckpt.pt" molecule_nat_lang "${NEW}/molecule_nat_lang/training_ready_hf_dataset"
run "molnatlang-medium-NEW" "${NEW}/molecule_nat_lang/gpt2-output/molecule_nat_lang-medium/ckpt.pt" molecule_nat_lang "${NEW}/molecule_nat_lang/training_ready_hf_dataset"
echo
echo "### genome medium (OLD vs NEW) ###"
run "genome-medium-OLD" "${OLD}/genome_sequence/gpt2-output/genome_sequence-medium/ckpt.pt" genome_sequence "${NEW}/genome_sequence/training_ready_hf_dataset" "$GENOME_SPM"
run "genome-medium-NEW" "${NEW}/genome_sequence/gpt2-output/genome_sequence-medium/ckpt.pt" genome_sequence "${NEW}/genome_sequence/training_ready_hf_dataset" "$GENOME_SPM"
echo
echo "==== finished at $(date) ===="
