#!/usr/bin/env bash
#SBATCH --job-name=eval-gpu-external-baselines
#SBATCH --partition=h200-long
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/eval_gpu_external_baselines_%j.log
#SBATCH --error=slurm_logs/eval_gpu_external_baselines_%j.log
#
# External HuggingFace baseline sweep on GPU. The first time this runs
# it downloads ChemBERTa-2 (~150MB), ESM-2 (~150MB), DNABERT-2 (~500MB)
# from huggingface_hub into ``$HF_HOME`` / ``~/.cache/huggingface/``.

set -uo pipefail
# shellcheck disable=SC1091
source workflows/common_functions.sh

export LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-/lustre/home/matsubara/riken-dataset-fundational-model/learning_source_20260316}"
export DEVICE="${DEVICE:-cuda}"

echo "[baseline-gpu] node: $(hostname)"
echo "[baseline-gpu] LSD:  $LEARNING_SOURCE_DIR"
nvidia-smi 2>/dev/null | head -10 || rocm-smi 2>/dev/null | head -10 || true
echo

bash workflows/eval-external-baselines.sh

echo
echo "[baseline-gpu] DONE."
