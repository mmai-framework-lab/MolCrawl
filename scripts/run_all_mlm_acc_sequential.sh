#!/bin/bash
# Run ALL 33 BERT mlm_accuracy evals **sequentially** in a single SLURM job
# on one GPU. Trades parallelism for queue priority — 33 separate jobs would
# get pushed 2 weeks out by FairShare under heavy cluster load, but a single
# 1-GPU job hits the queue once and runs all 33 ckpts back-to-back
# (per-model real GPU time is ~1-2 min).
#
# Use:
#   sbatch tmp/scripts/run_all_mlm_acc_sequential.sh
#
#SBATCH --partition=h200-long
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=mlm-acc-all
#SBATCH --output=/lustre/home/matsubara/learning_source_20260529_evo2species/genome_sequence/logs/mlm-acc-all-%j.log

set -e
REPO="/lustre/home/matsubara/riken-dataset-fundational-model"
OUT_ROOT="$REPO/tmp/mlm_accuracy"
PY="/lustre/home/matsubara/miniforge3/envs/molcrawl/bin/python"
mkdir -p "$OUT_ROOT"

cd "$REPO"

run_one() {
    local label="$1"
    local ckpt_dir="$2"
    local val_dir="$3"
    local modality="$4"
    local tok_dir="${5:-}"
    local out_dir="$OUT_ROOT/${label//\//__}"
    mkdir -p "$out_dir"
    if [ -f "$out_dir/mlm_accuracy.json" ]; then
        echo "[skip cached] $label"; return 0
    fi
    local extra=""
    [ -n "$tok_dir" ] && extra="--tokenizer-dir $tok_dir"
    echo ">>> $label"
    "$PY" tmp/scripts/eval_bert_mlm_accuracy.py \
        --ckpt-dir "$ckpt_dir" \
        --val-dir  "$val_dir" \
        --modality "$modality" \
        --arch bert \
        --label   "$label" \
        --out     "$out_dir" \
        $extra
}

latest_ckpt() {
    local d="$1"; [ -d "$d" ] || return 1
    ls "$d" 2>/dev/null | grep '^checkpoint-' | sort -V | tail -1
}

echo "=== start: $(date) ==="

# --- 通常 modality ---
declare -A MOD_SRC=(
    [protein_sequence]=/lustre/home/matsubara/learning_source_20260520_uncapped/protein_sequence
    [rna]=/lustre/home/matsubara/learning_source_20260316/rna
    [compounds]=/lustre/home/matsubara/learning_source_20260316/compounds
    [molecule_nat_lang]=/lustre/home/matsubara/learning_source_20260520_uncapped/molecule_nat_lang
)
declare -A MOD_VAL=(
    [protein_sequence]=/lustre/home/matsubara/learning_source_20260520_uncapped/protein_sequence/training_ready_hf_dataset
    [rna]=/lustre/home/matsubara/learning_source_20260520_uncapped/rna/training_ready_hf_dataset
    [compounds]=/lustre/home/matsubara/learning_source_20260316/compounds/chembl/training_ready_hf_dataset
    [molecule_nat_lang]=/lustre/home/matsubara/learning_source_20260520_uncapped/molecule_nat_lang/training_ready_hf_dataset
)

for mod in protein_sequence rna compounds molecule_nat_lang; do
    for sz in small medium large; do
        outdir="${MOD_SRC[$mod]}/bert-output/${mod}-${sz}"
        latest=$(latest_ckpt "$outdir") || continue
        [ -z "$latest" ] && { echo "[miss] $mod/$sz"; continue; }
        run_one "$mod/$sz" "$outdir/$latest" "${MOD_VAL[$mod]}" "$mod"
    done
done

# --- genome subset ---
EVO2=/lustre/home/matsubara/learning_source_20260529_evo2species/genome_sequence
TOK=$EVO2/custom_tokenizer_bert_single_nuc
for subset in mammal_centered \
              eukaryote_matched_random_seed{1..10} \
              global_random_seed{1..10}; do
    outdir="$EVO2/bert-output/genome_sequence-small-$subset"
    latest=$(latest_ckpt "$outdir") || continue
    [ -z "$latest" ] && { echo "[miss] genome/$subset"; continue; }
    val="$EVO2/$subset/training_ready_hf_dataset_bert"
    run_one "genome/$subset" "$outdir/$latest" "$val" "genome_sequence" "$TOK"
done

echo "=== done: $(date) ==="
ls -la "$OUT_ROOT"/*/mlm_accuracy.json | wc -l
echo "  → mlm_accuracy.json count above"
