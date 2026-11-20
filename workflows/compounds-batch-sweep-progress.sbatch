#!/bin/bash
#SBATCH --job-name=mc-cmp-progress
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Where each arm of the compounds batch-size grid has got to, for a progress report.
#
# No GPU: it reads the run logs. It runs here rather than on the login node because
# compute on the login node is not allowed here.
#
#   sbatch --export=ALL,ARMS=<...>/rt1-...tsv,LOGS=<log dir>,RUNS=<run output dir>,OUT=<...>/rt1-...tsv \
#          workflows/compounds-batch-sweep-progress.sbatch
#
# ARCH defaults to bert. A value passed through --export must not contain a comma:
# sbatch splits the list on commas and the value is truncated there.
set -uo pipefail

: "${ARMS:?set ARMS to the derived arm inventory TSV}"
: "${LOGS:?set LOGS to the directory holding the run logs}"
: "${RUNS:?set RUNS to the directory holding the run output directories}"
: "${OUT:?set OUT to the TSV to write}"
ARCH="${ARCH:-bert}"

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

echo "=== compounds batch-sweep progress on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "arch: ${ARCH}"

echo "########## 1. where each arm has got to ##########"
"${PYTHON}" scripts/compounds_batch_sweep_progress.py \
    --arms "${ARMS}" --logs "${LOGS}" --runs "${RUNS}" --arch "${ARCH}" --out "${OUT}"
rc=$?
echo "-- progress rc=${rc}"

echo
echo "########## 2. unit tests ##########"
"${PYTHON}" -m pytest -q tests/unit/test_compounds_batch_sweep_progress.py
trc=$?
echo "-- pytest rc=${trc}"

echo
echo "=== rc=$((rc + trc)) @ $(date -Is) ==="
exit $((rc + trc))
