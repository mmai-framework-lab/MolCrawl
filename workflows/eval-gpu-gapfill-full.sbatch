#!/usr/bin/env bash
#SBATCH --job-name=eval-gpu-gapfill-full
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/eval_gpu_gapfill_full_%j.log
#SBATCH --error=slurm_logs/eval_gpu_gapfill_full_%j.log
#
# Comprehensive GPU gap-fill: invokes ``workflows/eval-gapfill.sh`` with
# DEVICE=cuda so every remaining (modality × arch × size × task) combo
# that the CPU run can't handle (gpt2 ex-large OOM, slow proteingym ×
# bert/esm2 on CPU) lands a REPORT.md within minutes per combo.
#
# The driver is idempotent: combos whose REPORT.md already exists are
# skipped, so this sbatch can be re-submitted safely and will only do
# the work that remains.

set -uo pipefail
# shellcheck disable=SC1091
source workflows/common_functions.sh

echo "[gpu-gapfill-full] node: $(hostname)"
echo "[gpu-gapfill-full] Python: $PYTHON"
echo "[gpu-gapfill-full] LEARNING_SOURCE_DIR: $LEARNING_SOURCE_DIR"
nvidia-smi 2>/dev/null | head -10 || rocm-smi 2>/dev/null | head -10 || true
echo

# Bigger MAX (GPU is fast enough to afford larger sample sizes than the
# CPU gapfill default of 100) and more bootstrap resamples.
GAPFILL_DEVICE=cuda \
GAPFILL_MAX="${GAPFILL_MAX:-200}" \
GAPFILL_BOOTSTRAP="${GAPFILL_BOOTSTRAP:-100}" \
GAPFILL_FILTER="${GAPFILL_FILTER:-}" \
    bash workflows/eval-gapfill.sh

echo
echo "[gpu-gapfill-full] DONE. Rebuild the dashboard:"
echo "  bash workflows/eval-build-dashboard.sh"
