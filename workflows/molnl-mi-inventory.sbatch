#!/bin/bash
#SBATCH --job-name=mc-molnl-mi-inventory
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# 2026-11-10 judgment 3.3: count what the Mol-Instructions retrieval round left behind,
# and, with DELETE=1, remove exactly what the manifest from a counting run lists.
#
#   sbatch --export=ALL,REPORT_ROOT=<dir>,MANIFEST=<json> \
#          workflows/molnl-mi-inventory.sbatch
#   sbatch --export=ALL,MANIFEST=<json>,DELETE=1 workflows/molnl-mi-inventory.sbatch
#
# The deletion is a separate submission on purpose: the counting run has to be read by a
# person before anything is removed, and a flag that has to be set again is the cheapest
# way to keep those two apart.
#
# No GPU is needed; the partition allocates one anyway. It runs on a compute node because
# nothing in this project runs on a login node.
set -eo pipefail

: "${MANIFEST:?set MANIFEST to the manifest json to write or to read}"

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
cd "${REPO}"; mkdir -p workflows/slurm-logs "$(dirname "${MANIFEST}")"

if [ "${DELETE:-0}" = "1" ]; then
  echo "=== molnl Mol-Instructions deletion from ${MANIFEST} @ $(date -Is) ==="
  rc=0
  "${PYTHON}" scripts/molnl_mol_instructions_inventory.py \
    --manifest "${MANIFEST}" --delete || rc=$?
else
  : "${REPORT_ROOT:?set REPORT_ROOT to the report directory to scan}"
  [ -d "${REPORT_ROOT}" ] || { echo "REPORT_ROOT=${REPORT_ROOT} is not a directory" >&2; exit 1; }
  echo "=== molnl Mol-Instructions inventory of ${REPORT_ROOT} @ $(date -Is) ==="
  rc=0
  "${PYTHON}" scripts/molnl_mol_instructions_inventory.py \
    --report-root "${REPORT_ROOT}" --manifest "${MANIFEST}" || rc=$?
fi
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
