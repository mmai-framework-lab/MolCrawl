#!/bin/bash
#SBATCH --job-name=mc-zinc22-verify
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Read every fetched ZINC-22 file to its end, so a transfer spliced from several
# responses cannot pass for sound just because it is the right size.
#
#   sbatch --export=ALL,ZINC22_DEST=<dir>[,ZINC22_REPORT=<json>][,ZINC22_TRANCHES="H19 H20"] \
#          workflows/zinc22-verify-gzip.sbatch
#
# No network: everything this reads is already on disk.
set -uo pipefail
: "${ZINC22_DEST:?export ZINC22_DEST to the fetched tree}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== ZINC-22 gzip verify on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
# shellcheck disable=SC2086 - the tranche list is meant to split into arguments
"${PYTHON}" scripts/zinc22_verify_gzip.py --dest "${ZINC22_DEST}" \
    ${ZINC22_REPORT:+--report "${ZINC22_REPORT}"} \
    ${ZINC22_TRANCHES:+--tranches ${ZINC22_TRANCHES}}
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
