#!/bin/bash
#SBATCH --job-name=mc-cmp-trainval-curve
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Stitch the three compounds BERT stages into one train/validation curve table.
#
# No GPU: the script only reads log_history out of the furthest checkpoint of each
# of the twelve stage directories. It runs here rather than on the login node
# because that is twelve whole 45,000-step histories, and compute on the login node
# is not allowed here.
#
#   sbatch --export=ALL,BERT_ROOT=<...>/bert-output,CURVE_OUT=<...>/curve.tsv \
#          workflows/compounds-trainval-curve.sbatch
#
# Both roots are arguments and neither has a default: the run directories and the
# report directory are outside every checkout, and this repository is public.
set -uo pipefail

: "${BERT_ROOT:?set BERT_ROOT to the directory holding the compounds-*-seed42* runs}"
: "${CURVE_OUT:?set CURVE_OUT to the TSV to write}"
[ -d "${BERT_ROOT}" ] || { echo "BERT_ROOT=${BERT_ROOT} is not a directory" >&2; exit 1; }

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

echo "=== compounds train/valid curve on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:   ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "runs:   ${BERT_ROOT}"
echo "output: ${CURVE_OUT}"
"${PYTHON}" scripts/compounds_trainval_curve_tsv.py \
    --bert-root "${BERT_ROOT}" --out "${CURVE_OUT}"
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
