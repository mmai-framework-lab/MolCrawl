#!/bin/bash
#SBATCH --job-name=mc-reval-points
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Choose the five checkpoints per size the existing 2,560 arms are re-measured at,
# read the value each one already recorded, and -- once the eval-only jobs have run --
# collect their fixed-mask measurements into the same table.
#
# No GPU: it reads trainer_state.json out of the surviving checkpoints. It runs here
# rather than on the login node because compute on the login node is not allowed here.
#
#   sbatch --export=ALL,RUNS_ROOT=<.../compounds/bert-output>,POINTS_OUT=<...>/rt1-...tsv\
# [,COLLECT=<dir of eval-only records>] workflows/compounds-reval-points.sbatch
#
# A value passed through --export must not contain a comma: sbatch splits the list on
# commas and the value is truncated there.
set -uo pipefail

: "${RUNS_ROOT:?set RUNS_ROOT to the directory holding the output directories}"
: "${POINTS_OUT:?set POINTS_OUT to the TSV to write}"
COLLECT="${COLLECT:-}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
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

echo "=== compounds re-measurement points on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:    ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "runs:    ${RUNS_ROOT}"
echo "collect: ${COLLECT:-<nothing yet>}"
echo "output:  ${POINTS_OUT}"

rc=0
"${PYTHON}" scripts/compounds_reval_points.py \
    --runs-root "${RUNS_ROOT}" --collect "${COLLECT}" --out "${POINTS_OUT}"
step=$?; echo "-- points rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"

echo
echo "########## unit tests ##########"
"${PYTHON}" -m pytest -q tests/unit/test_compounds_reval_points.py \
    tests/unit/test_bert_eval_only.py
step=$?; echo "-- pytest rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"

echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
