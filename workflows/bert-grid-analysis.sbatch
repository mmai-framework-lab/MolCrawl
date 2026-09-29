#!/bin/bash
#SBATCH --job-name=mc-grid-analysis
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Reduce the BERT grid to one row per arm, then draw the relationships.
#
# No GPU is requested -- it reads trainer_state.json and safetensors headers --
# but compute does not belong on the login node here.
#
#   sbatch --export=ALL,GRID_CONFIG=<file.json>,GRID_TSV=<file.tsv>[,GRID_FIG_DIR=<dir>] \
#          workflows/bert-grid-analysis.sbatch
set -eo pipefail

: "${GRID_CONFIG:?set GRID_CONFIG to the JSON naming the run roots}"
: "${GRID_TSV:?set GRID_TSV to the TSV the summary is written into}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

export PYTHONPATH="${REPO}:${REPO}/scripts:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== grid analysis on ${SLURMD_NODENAME} @ $(date -Is) ==="
rc=0
"${PYTHON}" scripts/bert_grid_summary.py --config "${GRID_CONFIG}" --out "${GRID_TSV}" || rc=$?
if [ "${rc}" -eq 0 ] && [ -n "${GRID_FIG_DIR:-}" ]; then
  "${PYTHON}" scripts/plot_grid_analysis.py --summary "${GRID_TSV}" --out-dir "${GRID_FIG_DIR}" \
      --config "${GRID_CONFIG}" || rc=$?
fi
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
