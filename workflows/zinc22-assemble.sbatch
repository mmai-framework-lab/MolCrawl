#!/bin/bash
#SBATCH --job-name=mc-zinc22-assemble
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=workflows/slurm-logs/%x-%A_%a.out
#SBATCH --error=workflows/slurm-logs/%x-%A_%a.out
# Assign each shard's blocks to train/valid/test, one array task per shard. No task
# touches another's output, so there is no step left that has to see every shard.
#
#   ls <shards>/*.parquet > <list>
#   sbatch --array=1-2000%200 \
#          --export=ALL,ZINC22_SHARDLIST=<list>,ZINC22_DATASET=<dir>,ZINC22_SPLITCOUNTS=<dir> \
#          workflows/zinc22-assemble.sbatch
#
# More shards than MaxArraySize takes a second submission with ZINC22_OFFSET, the
# same way the packing array does.
#
# No network: everything this reads is already on disk.
set -uo pipefail
: "${ZINC22_SHARDLIST:?export ZINC22_SHARDLIST to the file holding one shard path per line}"
: "${ZINC22_DATASET:?export ZINC22_DATASET to the dataset root to write}"
: "${ZINC22_SPLITCOUNTS:?export ZINC22_SPLITCOUNTS to the directory for the per-shard counts}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs "${ZINC22_DATASET}" "${ZINC22_SPLITCOUNTS}"
IDX=$(( ${SLURM_ARRAY_TASK_ID:-1} + ${ZINC22_OFFSET:-0} ))
SHARD="$(sed -n "${IDX}p" "${ZINC22_SHARDLIST}")"
[ -n "${SHARD}" ] || { echo "no line ${IDX} in ${ZINC22_SHARDLIST}" >&2; exit 1; }
STEM="$(basename "${SHARD}" .parquet)"
echo "=== ZINC-22 assemble ${STEM} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
"${PYTHON}" scripts/zinc22_assemble.py \
    --shard "${SHARD}" --out "${ZINC22_DATASET}" \
    --counts "${ZINC22_SPLITCOUNTS}/${STEM}.json" \
    --valid-fraction "${ZINC22_VALID:-0.1}" --test-fraction "${ZINC22_TEST:-0.1}"
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
