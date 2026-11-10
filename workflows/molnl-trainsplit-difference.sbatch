#!/bin/bash
#SBATCH --job-name=mc-molnl-trainsplit-diff
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# 2026-11-10 judgment 3.2.1: the size of the gap between the pairs whose molecule occurs
# in train and the rest, from the json the split run wrote.
#
#   sbatch --export=ALL,SPLIT_JSON=<json>,OUT=<tsv> \
#          workflows/molnl-trainsplit-difference.sbatch
#
# Seconds of arithmetic over counts already measured, and it is a separate job from the
# scan so that re-reading the gap does not re-read 3.3M records. It runs on a compute node
# because nothing in this project runs on a login node.
set -eo pipefail

: "${SPLIT_JSON:?set SPLIT_JSON to the json scripts/molnl_smoltest_train_overlap_split.py wrote}"
: "${OUT:?set OUT to the tsv to write}"
[ -f "${SPLIT_JSON}" ] || { echo "SPLIT_JSON=${SPLIT_JSON} is not a file" >&2; exit 1; }

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

echo "=== molnl train-split difference on ${SLURMD_NODENAME} @ $(date -Is) ==="
rc=0
"${PYTHON}" scripts/molnl_trainsplit_difference.py \
  --split-json "${SPLIT_JSON}" --out "${OUT}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
