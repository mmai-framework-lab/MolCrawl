#!/bin/bash
#SBATCH --job-name=mc-run-inventory-page
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Build the run-inventory page from the published rows, a fresh run scan and the
# downstream scan.
#
# No GPU is requested -- it reads two TSVs, a JSON and a template -- but compute
# does not belong on the login node.
#
# The three inputs and the output live outside any checkout: they name run paths
# and this repository is public.
#
#   sbatch --export=ALL,PAGE_OLD=<json>,PAGE_RUNS=<tsv>,PAGE_DOWN=<tsv>,\
#PAGE_OUT=<html>,PAGE_ASOF=2026-10-29,PAGE_PREV=2026-09-01 \
#          workflows/run-inventory-page.sbatch
set -eo pipefail

: "${PAGE_OLD:?set PAGE_OLD to the JSON of previously published rows}"
: "${PAGE_RUNS:?set PAGE_RUNS to the run-inventory TSV}"
: "${PAGE_DOWN:?set PAGE_DOWN to the downstream-inventory TSV}"
: "${PAGE_OUT:?set PAGE_OUT to the HTML to write}"
: "${PAGE_ASOF:?set PAGE_ASOF to the as-of date}"
: "${PAGE_PREV:?set PAGE_PREV to the date of the previous stocktake}"
for f in "${PAGE_OLD}" "${PAGE_RUNS}" "${PAGE_DOWN}"; do
  [ -f "${f}" ] || { echo "${f} is not a file" >&2; exit 1; }
done

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

echo "=== run inventory page on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:  ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "out:   ${PAGE_OUT}"
rc=0
"${PYTHON}" scripts/build_run_inventory_page.py \
  --old-rows "${PAGE_OLD}" --new-tsv "${PAGE_RUNS}" --downstream "${PAGE_DOWN}" \
  --template scripts/templates/run-inventory.html --out "${PAGE_OUT}" \
  --asof "${PAGE_ASOF}" --previous "${PAGE_PREV}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
