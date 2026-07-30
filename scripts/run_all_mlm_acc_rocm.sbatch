#!/bin/bash
#SBATCH --partition=mi300x
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=mlm-acc-rocm
#SBATCH --output=/lustre/home/matsubara/learning_source_20260529_evo2species/genome_sequence/logs/mlm-acc-rocm-%j.log

set -e
cd /lustre/home/matsubara/riken-dataset-fundational-model

# common_functions.sh は AMD ノードで自動的に molcrawl_rocm を選択するが、
# 念のため明示的に固定 + ROCR_VISIBLE_DEVICES → CUDA_VISIBLE_DEVICES マッピング
PY="/lustre/home/matsubara/miniforge3/envs/molcrawl_rocm/bin/python"
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

echo "=== ROCm smoke ==="
echo "node: $(hostname)"
echo "PY  : $PY"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-unset}"
echo "ROCR_VISIBLE_DEVICES = ${ROCR_VISIBLE_DEVICES:-unset}"
"$PY" -c "
import torch
print(f'torch={torch.__version__}, hip={getattr(torch.version, \"hip\", None)}')
print(f'cuda available={torch.cuda.is_available()}, device count={torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'device 0: {torch.cuda.get_device_name(0)}')
"

# === 33 model sequential 実行 ===
OUT_ROOT="/lustre/home/matsubara/riken-dataset-fundational-model/tmp/mlm_accuracy"
mkdir -p "$OUT_ROOT"

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
ls -la "$OUT_ROOT"/*/mlm_accuracy.json 2>/dev/null | wc -l
echo "  ↑ mlm_accuracy.json count"
