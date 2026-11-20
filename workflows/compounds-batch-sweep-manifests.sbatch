#!/bin/bash
#SBATCH --job-name=mc-bsweep-manifests
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Read every arm's run_manifest.json and compare it with the derived plan, value by
# value (the 2026-11-20 directive's section 3).
#
# No GPU and no training data: it opens manifests and configs. It runs here rather
# than on the login node because compute on the login node is not allowed here.
#
#   sbatch --export=ALL,ARMS=<...>/rt1-...tsv,RUNS_ROOT=<runs root>,OUT=<...>/rt1-...tsv \
#          workflows/compounds-batch-sweep-manifests.sbatch
#
# A value passed through --export must not contain a comma: sbatch splits the list on
# commas and the value is truncated there.
set -uo pipefail

: "${ARMS:?set ARMS to the derived arm inventory TSV}"
: "${RUNS_ROOT:?set RUNS_ROOT to the runs root the arms wrote into}"
: "${OUT:?set OUT to the TSV to write}"

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

echo "=== compounds batch-sweep manifests on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:  ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "arms:  ${ARMS}"
echo "runs:  ${RUNS_ROOT}"
echo "out:   ${OUT}"

rc=0
echo
echo "########## 1. every arm's manifest against the plan ##########"
"${PYTHON}" scripts/compounds_batch_sweep_manifests.py \
    --arms "${ARMS}" --runs-root "${RUNS_ROOT}" --out "${OUT}"
step=$?; echo "-- manifests rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"

echo
echo "########## 2. unit tests ##########"
"${PYTHON}" -m pytest -q tests/unit/test_compounds_batch_sweep_manifests.py
step=$?; echo "-- pytest rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"

echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
