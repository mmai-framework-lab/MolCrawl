#!/bin/bash
#SBATCH --job-name=mc-zinc22-assemble
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Assemble the packed ZINC-22 shards into the train/valid/test layout the trainers
# load. Single task, because the split has to see every shard.
#
#   sbatch --export=ALL,ZINC22_SHARDS=<dir>,ZINC22_DATASET=<dir> \
#          workflows/zinc22-assemble.sbatch
#
# No network: everything this reads is already on disk.
set -uo pipefail
: "${ZINC22_SHARDS:?export ZINC22_SHARDS to the directory holding the parquet shards}"
: "${ZINC22_DATASET:?export ZINC22_DATASET to the dataset directory to write}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs "${ZINC22_DATASET}"
echo "=== ZINC-22 assemble on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "shards: $(ls -1 "${ZINC22_SHARDS}"/*.parquet 2>/dev/null | wc -l)"
"${PYTHON}" scripts/zinc22_assemble.py \
    --shards "${ZINC22_SHARDS}" --out "${ZINC22_DATASET}" \
    --valid-fraction "${ZINC22_VALID:-0.1}" --test-fraction "${ZINC22_TEST:-0.1}"
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
