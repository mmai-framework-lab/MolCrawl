#!/bin/bash
#SBATCH --job-name=mc-bsweep-arms
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Derive the compounds batch-size grid's arms and check the directive's own tables.
#
# No GPU and no data: it is arithmetic over the rules the 2026-11-19b directive
# states. It runs here rather than on the login node because compute on the login
# node is not allowed here.
#
#   sbatch --export=ALL,ARMS_OUT=<...>/rt1-...tsv[,ARMS_MICRO=8] \
#          workflows/compounds-batch-sweep-arms.sbatch
#
# ARMS_OUT has no default: the report directory is outside every checkout and this
# repository is public. ARMS_MICRO is one GPU's BERT micro-batch (default 32, which
# is what the existing compounds arms run at).
set -uo pipefail

: "${ARMS_OUT:?set ARMS_OUT to the TSV to write}"
MICRO="${ARMS_MICRO:-32}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
# A worktree holds only tracked files, so the interpreter is not inside it.
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== compounds batch-sweep arms on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:   ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "micro:  ${MICRO}"
echo "output: ${ARMS_OUT}"
"${PYTHON}" scripts/compounds_batch_sweep_arms.py \
    --bert-micro-batch "${MICRO}" --out "${ARMS_OUT}"
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
