#!/bin/bash
#SBATCH --job-name=mc-molnl-smoltest-summary
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# The 14 SMolInstruct test-split retrieval measurements on one sheet.
#
#   sbatch --export=ALL,RETRIEVAL_DIR=<dir>,OUT=<tsv>[,JOBS=<ids>] \
#          workflows/molnl-smoltest-summary.sbatch
#
# No GPU, and small enough to be mistaken for something that could run on a login node.
# It runs here because nothing in this project runs on a login node.
set -eo pipefail

: "${RETRIEVAL_DIR:?set RETRIEVAL_DIR to the directory holding retrieval-*.json}"
: "${OUT:?set OUT to the tsv to write}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs "$(dirname "${OUT}")"

echo "=== molnl smoltest summary on ${SLURMD_NODENAME} @ $(date -Is) ==="
rc=0
"${PYTHON}" scripts/molnl_smoltest_summary_tsv.py \
  --retrieval-dir "${RETRIEVAL_DIR}" --out "${OUT}" \
  ${JOBS:+--job "${JOBS}"} || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
