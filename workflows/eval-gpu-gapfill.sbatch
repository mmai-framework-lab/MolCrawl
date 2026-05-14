#!/usr/bin/env bash
#SBATCH --job-name=eval-gpu-gapfill
#SBATCH --partition=mi300x
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/eval_gpu_gapfill_%j.log
#SBATCH --error=slurm_logs/eval_gpu_gapfill_%j.log

set -uo pipefail
cd /lustre/home/matsubara/riken-dataset-fundational-model
# shellcheck disable=SC1091
source workflows/common_functions.sh

LSD="learning_source_20260316"
MATRIX_BASE="experiment_data/eval/matrix_v1"
DEVICE=cuda

echo "[gpu-gapfill] Python: $PYTHON"
echo "[gpu-gapfill] node: $(hostname)"
echo "[gpu-gapfill] GPU info:"
rocm-smi 2>/dev/null | head -20 || nvidia-smi 2>/dev/null | head -10 || true
echo

run_combo() {
    local label="$1" outdir="$2"
    shift 2
    if find "$outdir" -maxdepth 2 -name REPORT.md 2>/dev/null | grep -q .; then
        echo "[gpu-gapfill] SKIP (already done): $label"
        return 0
    fi
    mkdir -p "$outdir"
    echo "[gpu-gapfill] RUN: $label -> $outdir"
    OUTPUT_DIR="$outdir" DEVICE="$DEVICE" "$@" 2>&1 | tail -5 || {
        echo "[gpu-gapfill] FAILED: $label"
        return 1
    }
}

# 1) chembl_scaffold_heldout gpt2 ex-large (OOM on CPU; ~1.3B params)
ckpt="${LSD}/compounds_chembl/gpt2-output/compounds_chembl-ex-large/ckpt.pt"
[[ -f "$ckpt" ]] && run_combo "chembl_scaffold_heldout__gpt2__ex-large" \
    "${MATRIX_BASE}/chembl_scaffold_heldout_gpt2_ex-large" \
    bash -c "MODEL_PATH='${ckpt}' \
             TOKENIZER_PATH='assets/molecules/vocab.txt' \
             HELDOUT_CSV='${LSD}/eval/chembl_scaffold_heldout/heldout.csv' \
             MAX_EXAMPLES=200 SEED=42 PREDICTIONS_PREVIEW_COUNT=10 \
             bash workflows/eval-chembl-heldout.sh"

# 2) cosmic gpt2 ex-large
ckpt="${LSD}/genome_sequence/gpt2-output/genome_sequence-ex-large/ckpt.pt"
cosmic_csv="${LSD}/eval/cosmic/cosmic_eval.csv"
if [[ -f "$ckpt" && -f "$cosmic_csv" ]]; then
    run_combo "cosmic__gpt2__ex-large" \
        "${MATRIX_BASE}/cosmic_gpt2_ex-large" \
        bash -c "MODEL_PATH='${ckpt}' \
                 TOKENIZER_PATH='${LSD}/genome_sequence/spm_tokenizer.model' \
                 COSMIC_DATA='${cosmic_csv}' \
                 ARCH=gpt2 N_PER_CLASS=50 PREDICTIONS_PREVIEW_COUNT=8 \
                 BOOTSTRAP_SAMPLES=100 \
                 bash workflows/eval-cosmic.sh"
fi

# 3) molecule_nat_lang gpt2 ex-large
ckpt="${LSD}/molecule_nat_lang/gpt2-output/molecule_nat_lang-ex-large/ckpt.pt"
pairs="${LSD}/eval/molecule_nat_lang/pairs.csv"
if [[ -f "$ckpt" && -f "$pairs" ]]; then
    run_combo "molecule_nat_lang__gpt2__ex-large" \
        "${MATRIX_BASE}/molecule_nat_lang_gpt2_ex-large" \
        bash -c "MODEL_PATH='${ckpt}' \
                 PAIRS_CSV='${pairs}' \
                 ARCH=gpt2 MAX_EXAMPLES=200 PREDICTIONS_PREVIEW_COUNT=8 \
                 bash workflows/eval-molecule-nat-lang.sh"
fi

# 4) chemllmbench gpt2 large (CPU was estimated 60-90 min; GPU should be 5-10 min)
ckpt="${LSD}/molecule_nat_lang/gpt2-output/molecule_nat_lang-large/ckpt.pt"
chemllm_dir="${LSD}/eval/chemllmbench"
if [[ -f "$ckpt" && -d "$chemllm_dir" ]]; then
    run_combo "chemllmbench__gpt2__large" \
        "${MATRIX_BASE}/chemllmbench_gpt2_large" \
        bash -c "MODEL_PATH='${ckpt}' \
                 CHEMLLMBENCH_DIR='${chemllm_dir}' \
                 SUBTASKS='molecule_captioning molecule_design reaction_prediction name_conversion retrosynthesis yield_prediction property_prediction' \
                 ARCH=gpt2 MAX_EXAMPLES=30 \
                 MAX_NEW_TOKENS=24 PREDICTIONS_PREVIEW_COUNT=8 \
                 bash workflows/eval-chemllmbench.sh"
fi

# 5) chemllmbench gpt2 ex-large (will use molecule_nat_lang ex-large ckpt — only fits on GPU)
ckpt="${LSD}/molecule_nat_lang/gpt2-output/molecule_nat_lang-ex-large/ckpt.pt"
if [[ -f "$ckpt" && -d "$chemllm_dir" ]]; then
    run_combo "chemllmbench__gpt2__ex-large" \
        "${MATRIX_BASE}/chemllmbench_gpt2_ex-large" \
        bash -c "MODEL_PATH='${ckpt}' \
                 CHEMLLMBENCH_DIR='${chemllm_dir}' \
                 SUBTASKS='molecule_captioning molecule_design reaction_prediction name_conversion retrosynthesis yield_prediction property_prediction' \
                 ARCH=gpt2 MAX_EXAMPLES=30 \
                 MAX_NEW_TOKENS=24 PREDICTIONS_PREVIEW_COUNT=8 \
                 bash workflows/eval-chemllmbench.sh"
fi

echo "[gpu-gapfill] DONE all combos"
