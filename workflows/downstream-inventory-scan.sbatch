#!/bin/bash
#SBATCH --job-name=mc-downstream-inventory
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Read the downstream result files and write one row per evaluation x model.
#
# No GPU is requested -- it reads result JSONs and TSVs -- but compute does not
# belong on the login node, and the ProteinGym reader walks the per-assay
# metrics tree on Lustre.
#
# The config names the result directories and the output goes outside any
# checkout: both name run paths and this repository is public.
#
#   sbatch --export=ALL,DS_CONFIG=<file.json>,DS_OUT=<file.tsv> \
#          workflows/downstream-inventory-scan.sbatch
set -eo pipefail

: "${DS_CONFIG:?set DS_CONFIG to the JSON naming the result directories}"
: "${DS_OUT:?set DS_OUT to the TSV the inventory is written into}"
[ -f "${DS_CONFIG}" ] || { echo "DS_CONFIG=${DS_CONFIG} is not a file" >&2; exit 1; }

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

echo "=== downstream inventory on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:    ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "config:  ${DS_CONFIG}"
echo "out:     ${DS_OUT}"
rc=0
"${PYTHON}" scripts/downstream_inventory_scan.py --config "${DS_CONFIG}" --out "${DS_OUT}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
