#!/bin/bash
#SBATCH --job-name=remaining-gpt2-tierA
#SBATCH --partition=h200-long
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/remaining-gpt2-tierA-%j.log
#SBATCH --error=slurm_logs/remaining-gpt2-tierA-%j.log

# Tier A perplexity for every gpt2 checkpoint that finished retraining on the
# NEW corpus (LEARNING_SOURCE_DIR) but has not yet been evaluated. Same
# OLD-vs-NEW per-modality contract as eval-medium-gpt2-tier-a.sh — pulls OLD
# baselines from OLD_LEARNING_SOURCE_DIR and evaluates both against the NEW
# modality's own valid split, capped at 2000 samples.

set -e
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the NEW (uncapped) corpus root before submitting}"
export LEARNING_SOURCE_DIR
PYTHON="${PYTHON:-python}"
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-}"

# NEW corpus = LEARNING_SOURCE_DIR; the OLD corpus used for the OLD-vs-NEW
# comparison is supplied separately via OLD_LEARNING_SOURCE_DIR.
NEW="${LEARNING_SOURCE_DIR}"
: "${OLD_LEARNING_SOURCE_DIR:?set OLD_LEARNING_SOURCE_DIR to the OLD corpus root for the OLD-vs-NEW comparison}"
OLD="${OLD_LEARNING_SOURCE_DIR}"
REPORT="${NEW}/protein_sequence/logs/remaining-gpt2-tierA-${SLURM_JOB_ID}"
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

echo "==== remaining gpt2 Tier A job ${SLURM_JOB_ID} at $(date) ===="

# --- protein large / xl (small/medium already in earlier reports) ---
echo
echo "### protein large / xl (OLD vs NEW) ###"
for sz in large ex-large ; do
    run "protein-${sz}-OLD" "${OLD}/protein_sequence/gpt2-output/protein_sequence-${sz}/ckpt.pt" protein_sequence "${NEW}/protein_sequence/training_ready_hf_dataset"
    run "protein-${sz}-NEW" "${NEW}/protein_sequence/gpt2-output/protein_sequence-${sz}/ckpt.pt" protein_sequence "${NEW}/protein_sequence/training_ready_hf_dataset"
done

# --- genome large / xl (small/medium already in earlier reports) ---
echo
echo "### genome large / xl (OLD vs NEW) ###"
for sz in large ex-large ; do
    run "genome-${sz}-OLD" "${OLD}/genome_sequence/gpt2-output/genome_sequence-${sz}/ckpt.pt" genome_sequence "${NEW}/genome_sequence/training_ready_hf_dataset" "$GENOME_SPM"
    run "genome-${sz}-NEW" "${NEW}/genome_sequence/gpt2-output/genome_sequence-${sz}/ckpt.pt" genome_sequence "${NEW}/genome_sequence/training_ready_hf_dataset" "$GENOME_SPM"
done

# --- compounds main (all four sizes never reported before) ---
echo
echo "### compounds main (OLD vs NEW) ###"
for sz in small medium large ex-large ; do
    run "compounds-${sz}-OLD" "${OLD}/compounds/gpt2-output/compounds-${sz}/ckpt.pt" compounds "${NEW}/compounds/training_ready_hf_dataset"
    run "compounds-${sz}-NEW" "${NEW}/compounds/gpt2-output/compounds-${sz}/ckpt.pt" compounds "${NEW}/compounds/training_ready_hf_dataset"
done

# --- compounds chembl FT (post-bugfix retraining) ---
echo
echo "### compounds_chembl (OLD vs NEW) ###"
for sz in small medium large ex-large ; do
    run "compounds_chembl-${sz}-OLD" "${OLD}/compounds_chembl/gpt2-output/compounds_chembl-${sz}/ckpt.pt" compounds "${NEW}/compounds/chembl/training_ready_hf_dataset"
    run "compounds_chembl-${sz}-NEW" "${NEW}/compounds_chembl/gpt2-output/compounds_chembl-${sz}/ckpt.pt" compounds "${NEW}/compounds/chembl/training_ready_hf_dataset"
done

# --- compounds guacamol FT (dataset lives under compounds/benchmark/GuacaMol/) ---
GUACAMOL_DS="${NEW}/compounds/benchmark/GuacaMol/compounds/training_ready_hf_dataset"
echo
echo "### compounds_guacamol (OLD vs NEW) ###"
for sz in small medium large ex-large ; do
    run "compounds_guacamol-${sz}-OLD" "${OLD}/compounds_guacamol/gpt2-output/compounds_guacamol-${sz}/ckpt.pt" compounds "${GUACAMOL_DS}"
    run "compounds_guacamol-${sz}-NEW" "${NEW}/compounds_guacamol/gpt2-output/compounds_guacamol-${sz}/ckpt.pt" compounds "${GUACAMOL_DS}"
done

# --- rna (all four sizes) ---
echo
echo "### rna (OLD vs NEW) ###"
for sz in small medium large ex-large ; do
    run "rna-${sz}-OLD" "${OLD}/rna/gpt2-output/rna-${sz}/ckpt.pt" rna "${NEW}/rna/training_ready_hf_dataset"
    run "rna-${sz}-NEW" "${NEW}/rna/gpt2-output/rna-${sz}/ckpt.pt" rna "${NEW}/rna/training_ready_hf_dataset"
done

echo
echo "==== finished at $(date) ===="
echo "Report dir: ${REPORT}"
