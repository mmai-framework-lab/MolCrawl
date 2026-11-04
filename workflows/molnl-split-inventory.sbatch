#!/bin/bash
#SBATCH --job-name=molnl-split-inv
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# What is in each molecule_nat_lang split, and which of it any run has read.
#
# No GPU is requested -- it reads arrow metadata and sums one column -- but summing over
# 3.3M rows does not belong on the login node.
#
#   sbatch --export=ALL,ARROW_ROOT=<arrow_splits>,PACKED=<training_ready_hf_dataset>,\
#          OUT_JSON=<json>,OUT_TSV=<tsv> workflows/molnl-split-inventory.sbatch
#
# Optional: SPLITS (default train,valid,test), PYTHON.
set -eo pipefail

: "${ARROW_ROOT:?set ARROW_ROOT to the directory holding train.arrow / valid.arrow / test.arrow}"
: "${PACKED:?set PACKED to the packed training_ready_hf_dataset the runs read}"
[ -d "${ARROW_ROOT}" ] || { echo "ARROW_ROOT=${ARROW_ROOT} is not a directory" >&2; exit 1; }
[ -d "${PACKED}" ] || { echo "PACKED=${PACKED} is not a directory" >&2; exit 1; }

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
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== split inventory on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "=== records: ${ARROW_ROOT} ==="
echo "=== packed:  ${PACKED} ==="
rc=0
"${PYTHON}" scripts/molnl_split_inventory.py \
  --arrow-root "${ARROW_ROOT}" --packed "${PACKED}" \
  --splits "${SPLITS:-train,valid,test}" \
  ${OUT_JSON:+--out-json "${OUT_JSON}"} \
  ${OUT_TSV:+--out-tsv "${OUT_TSV}"} || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
