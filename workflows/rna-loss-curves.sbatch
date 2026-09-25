#!/bin/bash
# Draw the loss curves of the grid, and write the numbers they were drawn from.
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --job-name=rna-loss-curves
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${RUNS_ROOT:?export RUNS_ROOT to the output root of the grid}"
: "${CURVE_OUT_DIR:?export CURVE_OUT_DIR to where the TSV and figures should go}"
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 MPLCONFIGDIR="${CURVE_OUT_DIR}/.mpl"
mkdir -p workflows/slurm-logs "${CURVE_OUT_DIR}" "${MPLCONFIGDIR}"
echo "=== rna-loss-curves @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
"${PYROOT}/miniconda/bin/python" scripts/rna_loss_curves.py \
  --runs-root "${RUNS_ROOT}" --out-dir "${CURVE_OUT_DIR}" \
  --arch "${ARCH:-gpt2}" ${METRIC:+--metric "${METRIC}"}
