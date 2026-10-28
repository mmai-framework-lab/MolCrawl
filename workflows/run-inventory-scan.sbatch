#!/bin/bash
#SBATCH --job-name=mc-run-inventory
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Walk the run trees and write one row per training run.
#
# No GPU is requested -- it stats directories and reads run_manifest.json -- but
# compute does not belong on the login node here, and these trees hold tens of
# thousands of entries on Lustre.
#
# The config names the trees and the output goes outside any checkout: both name
# run paths and this repository is public.
#
#   sbatch --export=ALL,INV_CONFIG=<file.json>,INV_OUT=<file.tsv> \
#          workflows/run-inventory-scan.sbatch
set -eo pipefail

: "${INV_CONFIG:?set INV_CONFIG to the JSON naming the trees to walk}"
: "${INV_OUT:?set INV_OUT to the TSV the inventory is written into}"
[ -f "${INV_CONFIG}" ] || { echo "INV_CONFIG=${INV_CONFIG} is not a file" >&2; exit 1; }

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
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

echo "=== run inventory on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:    ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "config:  ${INV_CONFIG}"
echo "out:     ${INV_OUT}"
rc=0
"${PYTHON}" scripts/run_inventory_scan.py --config "${INV_CONFIG}" --out "${INV_OUT}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
