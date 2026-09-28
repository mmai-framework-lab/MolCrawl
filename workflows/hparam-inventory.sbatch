#!/bin/bash
#SBATCH --job-name=mc-hparam-inventory
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Read the pretrain configs and write the hyperparameter table on a compute node.
#
# No GPU is requested -- it only parses Python files -- but compute does not
# belong on the login node here.
#
#   sbatch --export=ALL,HPARAM_OUT=<file.tsv>[,HPARAM_ARCH=bert|gpt2] \
#          workflows/hparam-inventory.sbatch
set -eo pipefail

: "${HPARAM_OUT:?set HPARAM_OUT to the TSV the table is written into}"

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
export PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== hyperparameter inventory on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:   ${REPO}  $(git -C "${REPO}" rev-parse --abbrev-ref HEAD) $(git -C "${REPO}" rev-parse --short HEAD)"
echo "out:    ${HPARAM_OUT}"
rc=0
"${PYTHON}" scripts/hparam_inventory.py --out "${HPARAM_OUT}" ${HPARAM_ARCH:+--arch "${HPARAM_ARCH}"} || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
