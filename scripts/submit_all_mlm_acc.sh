#!/bin/bash
# Submit one SLURM job per (modality, size) BERT ckpt to compute MLM accuracy.
# Each job calls tmp/scripts/eval_bert_mlm_accuracy.py for one ckpt only,
# so they run in parallel as GPUs become available.
#
# Output: $OUT_ROOT/<label>/mlm_accuracy.json
set -e

REPO="/lustre/home/matsubara/riken-dataset-fundational-model"
OUT_ROOT="$REPO/tmp/mlm_accuracy"
mkdir -p "$OUT_ROOT"

# ---- helper: pick the highest-numbered checkpoint-NNNNN under a dir ----
latest_ckpt() {
    local d="$1"
    [ -d "$d" ] || return 1
    ls "$d" 2>/dev/null | grep '^checkpoint-' | sort -V | tail -1
}

# ---- helper: submit one SLURM job ----
submit_one() {
    local label="$1"
    local ckpt_dir="$2"
    local val_dir="$3"
    local modality="$4"
    local extra_tokenizer="${5:-}"
    local out_dir="$OUT_ROOT/${label//\//__}"
    mkdir -p "$out_dir"

    if [ -f "$out_dir/mlm_accuracy.json" ]; then
        echo "SKIP (cached): $label"
        return 0
    fi

    local extra_args=""
    [ -n "$extra_tokenizer" ] && extra_args="--tokenizer-dir $extra_tokenizer"

    echo "SUBMIT: $label"
    sbatch \
        --partition=h200-long \
        --gres=gpu:1 \
        --time=02:00:00 \
        --job-name="mlm-${label//\//-}" \
        --output="$out_dir/slurm.log" \
        --wrap="cd $REPO && \
                /lustre/home/matsubara/miniforge3/envs/molcrawl/bin/python \
                  tmp/scripts/eval_bert_mlm_accuracy.py \
                  --ckpt-dir '$ckpt_dir' \
                  --val-dir '$val_dir' \
                  --modality '$modality' \
                  --arch bert \
                  --label '$label' \
                  --out '$out_dir' \
                  $extra_args"
}

# ---- 通常 modality (canonical source 優先で 1 つ) ----
declare -A MOD_CKPT_SRC=(
    [protein_sequence]="/lustre/home/matsubara/learning_source_20260520_uncapped/protein_sequence"
    [rna]="/lustre/home/matsubara/learning_source_20260316/rna"
    [compounds]="/lustre/home/matsubara/learning_source_20260316/compounds"
    [molecule_nat_lang]="/lustre/home/matsubara/learning_source_20260520_uncapped/molecule_nat_lang"
)
declare -A MOD_VAL_DIR=(
    [protein_sequence]="/lustre/home/matsubara/learning_source_20260520_uncapped/protein_sequence/training_ready_hf_dataset"
    [rna]="/lustre/home/matsubara/learning_source_20260520_uncapped/rna/training_ready_hf_dataset"
    [compounds]="/lustre/home/matsubara/learning_source_20260316/compounds/chembl/training_ready_hf_dataset"
    [molecule_nat_lang]="/lustre/home/matsubara/learning_source_20260520_uncapped/molecule_nat_lang/training_ready_hf_dataset"
)

for mod in protein_sequence rna compounds molecule_nat_lang; do
    src="${MOD_CKPT_SRC[$mod]}"
    val="${MOD_VAL_DIR[$mod]}"
    for size in small medium large; do
        outdir="$src/bert-output/${mod}-${size}"
        latest=$(latest_ckpt "$outdir")
        if [ -z "$latest" ]; then
            echo "MISS: $mod/$size (no checkpoint-* under $outdir)"
            continue
        fi
        submit_one "$mod/$size" "$outdir/$latest" "$val" "$mod"
    done
done

# ---- genome_sequence Evo2 subset (21 subset, BERT, small のみ) ----
EVO2="/lustre/home/matsubara/learning_source_20260529_evo2species/genome_sequence"
TOK="$EVO2/custom_tokenizer_bert_single_nuc"
for subset in mammal_centered \
              eukaryote_matched_random_seed{1..10} \
              global_random_seed{1..10}; do
    outdir="$EVO2/bert-output/genome_sequence-small-$subset"
    latest=$(latest_ckpt "$outdir")
    if [ -z "$latest" ]; then
        echo "MISS: genome/$subset (no checkpoint-* under $outdir)"
        continue
    fi
    val="$EVO2/$subset/training_ready_hf_dataset_bert"
    submit_one "genome/$subset" "$outdir/$latest" "$val" "genome_sequence" "$TOK"
done

echo
echo "=== queue ==="
squeue -u matsubara | head -40
