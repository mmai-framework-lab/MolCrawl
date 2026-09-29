#!/bin/bash
#SBATCH --job-name=mc-globalbatch-check
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Check every pretrain config's declared global batch against its own factors.
#
# No GPU is requested -- it parses Python files -- but compute does not belong on
# the login node here.
#
#   sbatch [--export=ALL,GB_WORLD_SIZE=4] workflows/check-global-batch.sbatch
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

echo "=== global batch declarations on ${SLURMD_NODENAME} @ $(date -Is) ==="
rc=0
"${PYTHON}" scripts/check_global_batch_declarations.py --world-size "${GB_WORLD_SIZE:-4}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
