#!/bin/bash
#SBATCH --job-name=mc-count-packed
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Count the molecules a packed corpus actually holds, per split, by counting the
# separators. A count taken from the train split alone is 80 per cent of the corpus.
#
#   sbatch --export=ALL,PACKED_DATASET=<dir>,PACKED_SEP_ID=13[,PACKED_REPORT=<json>] \
#          workflows/count-packed-molecules.sbatch
#
# No network and no GPU: everything this reads is already on disk.
set -uo pipefail
: "${PACKED_DATASET:?export PACKED_DATASET to the DatasetDict directory}"
: "${PACKED_SEP_ID:?export PACKED_SEP_ID to the separator token id}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== packed molecule count on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "dataset: ${PACKED_DATASET}"
"${PYTHON}" scripts/count_packed_molecules.py \
    --dataset "${PACKED_DATASET}" --sep-id "${PACKED_SEP_ID}" \
    ${PACKED_REPORT:+--report "${PACKED_REPORT}"}
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
