#!/usr/bin/env bash
#SBATCH --job-name=eval-gpu-gapfill-finetune
#SBATCH --partition=h200-long
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/eval_gpu_gapfill_finetune_%j.log
#SBATCH --error=slurm_logs/eval_gpu_gapfill_finetune_%j.log
#
# Fine-tune evaluation gap-fill: invokes ``workflows/eval-gapfill-finetune.sh``
# with DEVICE=cuda so every (corpus, arch, size, task) combo whose
# fine-tune checkpoint exists is evaluated on GPU. Idempotent — combos
# whose REPORT.md already exists are skipped.
#
# Uses h200-long (24h timelimit) so the job survives even if the queue
# is busy or runs slower than expected. Resubmittable safely.

set -uo pipefail
# shellcheck disable=SC1091
source workflows/common_functions.sh

export LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-/lustre/home/matsubara/riken-dataset-fundational-model/learning_source_20260316}"

echo "[ft-gapfill-gpu] node: $(hostname)"
echo "[ft-gapfill-gpu] Python: $PYTHON"
echo "[ft-gapfill-gpu] LEARNING_SOURCE_DIR: $LEARNING_SOURCE_DIR"
nvidia-smi 2>/dev/null | head -10 || rocm-smi 2>/dev/null | head -10 || true
echo

GAPFILL_DEVICE=cuda \
GAPFILL_MAX="${GAPFILL_MAX:-200}" \
GAPFILL_BOOTSTRAP="${GAPFILL_BOOTSTRAP:-50}" \
GAPFILL_ONLY_ARCH="${GAPFILL_ONLY_ARCH:-gpt2}" \
GAPFILL_FILTER="${GAPFILL_FILTER:-}" \
    bash workflows/eval-gapfill-finetune.sh

echo
echo "[ft-gapfill-gpu] DONE. Rebuild the dashboard:"
echo "  bash workflows/eval-build-dashboard.sh"
