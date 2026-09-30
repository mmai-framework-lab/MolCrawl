#!/bin/bash
#SBATCH --job-name=mc-seed-inventory
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Report the seed every recent run actually used.
#
# No GPU is requested -- it parses Python files -- but compute does not belong on
# the login node here.
#
#   sbatch --export=ALL,SEED_ROOTS="<dir> <dir>"[,SEED_DAYS=2] workflows/seed-inventory-runs.sbatch
set -eo pipefail

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

export PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== seed declarations on ${SLURMD_NODENAME} @ $(date -Is) ==="
rc=0
"${PYTHON}" scripts/seed_inventory_runs.py --roots ${SEED_ROOTS} --since-days "${SEED_DAYS:-2}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
