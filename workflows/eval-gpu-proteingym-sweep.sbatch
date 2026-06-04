#!/usr/bin/env bash
#SBATCH --job-name=eval-gpu-proteingym-sweep
#SBATCH --partition=h200-long
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/eval_gpu_proteingym_sweep_%j.log
#SBATCH --error=slurm_logs/eval_gpu_proteingym_sweep_%j.log
#
# ProteinGym full-assay sweep wrapper. Defaults to the
# ``protein_sequence × gpt2 × medium`` model so a single sbatch run
# produces the canonical 217-assay ProteinGym table. Override with
# MODEL_PATH / MODEL_SLUG / ARCH if you want a different combo.

set -uo pipefail
# shellcheck disable=SC1091
source workflows/common_functions.sh

export LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-/lustre/home/matsubara/riken-dataset-fundational-model/learning_source_20260316}"

# Defaults: protein × gpt2 × medium (the canonical ProteinGym sweep target).
export MODEL_PATH="${MODEL_PATH:-${LEARNING_SOURCE_DIR}/protein_sequence/gpt2-output/protein_sequence-medium/ckpt.pt}"
export MODEL_SLUG="${MODEL_SLUG:-protein_sequence-gpt2-medium}"
export ARCH="${ARCH:-gpt2}"
export MODALITY="${MODALITY:-protein_sequence}"
export DEVICE="${DEVICE:-cuda}"
export BOOTSTRAP="${BOOTSTRAP:-100}"
export MAX_EXAMPLES="${MAX_EXAMPLES:-200}"

echo "[pg-sweep-gpu] node:        $(hostname)"
echo "[pg-sweep-gpu] LSD:         ${LEARNING_SOURCE_DIR}"
echo "[pg-sweep-gpu] MODEL_PATH:  ${MODEL_PATH}"
echo "[pg-sweep-gpu] MODEL_SLUG:  ${MODEL_SLUG}"
nvidia-smi 2>/dev/null | head -10 || rocm-smi 2>/dev/null | head -10 || true
echo

bash workflows/eval-proteingym-sweep.sh

echo
echo "[pg-sweep-gpu] DONE. Rebuild dashboard:"
echo "  bash workflows/eval-build-dashboard.sh"
