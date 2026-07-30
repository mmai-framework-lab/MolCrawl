#!/bin/bash
#SBATCH --partition=mi300x
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=tba-mlm-acc
#SBATCH --output=/lustre/home/matsubara/learning_source_20260316/genome_sequence/logs/tba-mlm-acc-%j.log

set -e
cd /lustre/home/matsubara/riken-dataset-fundational-model

PY="/lustre/home/matsubara/miniforge3/envs/molcrawl_rocm/bin/python"
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

echo "=== ROCm smoke ==="
echo "node: $(hostname)"
"$PY" -c "
import torch
print(f'torch={torch.__version__}, hip={getattr(torch.version, \"hip\", None)}')
print(f'cuda available={torch.cuda.is_available()}, device count={torch.cuda.device_count()}')
"

OUT_ROOT="/lustre/home/matsubara/riken-dataset-fundational-model/tmp/mlm_accuracy"
mkdir -p "$OUT_ROOT"

run_one() {
    local label="$1"
    local ckpt_dir="$2"
    local val_dir="$3"
    local tok_dir="$4"
    local arch="${5:-bert}"
    local out_dir="$OUT_ROOT/${label//\//__}"
    mkdir -p "$out_dir"
    if [ -f "$out_dir/mlm_accuracy.json" ]; then
        echo "[skip cached] $label"; return 0
    fi
    echo ">>> $label"
    "$PY" tmp/scripts/eval_bert_mlm_accuracy.py \
        --ckpt-dir "$ckpt_dir" \
        --val-dir  "$val_dir" \
        --modality genome_sequence \
        --arch     "$arch" \
        --label    "$label" \
        --out      "$out_dir" \
        --tokenizer-dir "$tok_dir"
}

GS=/lustre/home/matsubara/learning_source_20260316/genome_sequence
VAL=$GS/training_ready_hf_dataset

echo "=== start: $(date) ==="
run_one "genome_seq_tba/bert-small-20260316"    \
    "$GS/bert-output/genome_sequence-small/checkpoint-60000" \
    "$VAL" "$GS/custom_tokenizer_bert" "bert"
run_one "genome_seq_tba/dnabert2-medium-20260316" \
    "$GS/dnabert2-output/dnabert2-medium/checkpoint-200000" \
    "$VAL" "$GS/custom_tokenizer_dnabert2" "bert"
run_one "genome_seq_tba/dnabert2-large-20260316"  \
    "$GS/dnabert2-output/dnabert2-large/checkpoint-200000" \
    "$VAL" "$GS/custom_tokenizer_dnabert2" "bert"
echo "=== done: $(date) ==="
ls -la "$OUT_ROOT"/genome_seq_tba__*/mlm_accuracy.json 2>/dev/null
