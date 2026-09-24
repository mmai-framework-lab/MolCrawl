#!/bin/bash
#SBATCH --job-name=mc-plot-loss
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Plot the eval-loss series of a set of runs on a compute node.
#
# No GPU is requested -- it only reads logs and trainer_state.json and writes PNGs
# -- but compute does not belong on the login node here, and reading several
# hundred megabytes of run logs is not something to do there either.
#
# The config and the output directory are BOTH outside any checkout: they name run
# paths, and this repository is public. Neither is written into this file.
#
#   sbatch --export=ALL,LOSS_CONFIG=<file.json>,LOSS_OUT_DIR=<dir> \
#          workflows/plot-eval-loss-series.sbatch
set -eo pipefail

: "${LOSS_CONFIG:?set LOSS_CONFIG to the JSON config naming the runs to plot}"
: "${LOSS_OUT_DIR:?set LOSS_OUT_DIR to the directory the PNGs are written into}"
[ -f "${LOSS_CONFIG}" ] || { echo "LOSS_CONFIG=${LOSS_CONFIG} is not a file" >&2; exit 1; }

REPO="${SLURM_SUBMIT_DIR:-$PWD}"

# A git worktree holds only tracked files, so the interpreter is not inside it.
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== loss plots on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:    ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "config:  ${LOSS_CONFIG}"
echo "outputs: ${LOSS_OUT_DIR}"
rc=0
"${PYTHON}" scripts/plot_eval_loss_series.py \
    --config "${LOSS_CONFIG}" --out-dir "${LOSS_OUT_DIR}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
