#!/bin/bash
#SBATCH --job-name=mc-run-report-pages
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Render one page per run from its run_manifest.json.
#
# No GPU is requested -- it opens one JSON per run -- but compute does not
# belong on the login node and the manifests sit in the run trees on Lustre.
#
# The inputs and the output live outside any checkout: they name run paths and
# this repository is public.
#
#   sbatch --export=ALL,RP_ROOTS=<config.json>,RP_RUNS=<tsv>,RP_HEALTH=<json>,\
#RP_DOWN=<tsv>,RP_OUT=<dir>,RP_INDEX=<json>,RP_ASOF=2026-10-29,RP_LIMIT=30 \
#          workflows/run-report-pages.sbatch
set -eo pipefail

: "${RP_ROOTS:?set RP_ROOTS to the run-inventory config naming the trees}"
: "${RP_RUNS:?set RP_RUNS to the run-inventory TSV}"
: "${RP_HEALTH:?set RP_HEALTH to the checkpoint-metric audit JSON}"
: "${RP_DOWN:?set RP_DOWN to the downstream-inventory TSV}"
: "${RP_OUT:?set RP_OUT to the directory the pages are written into}"
: "${RP_INDEX:?set RP_INDEX to the JSON mapping runs to their page}"
: "${RP_ASOF:?set RP_ASOF to the as-of date}"
for f in "${RP_ROOTS}" "${RP_RUNS}" "${RP_HEALTH}" "${RP_DOWN}"; do
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

echo "=== run report pages on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:  ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "out:   ${RP_OUT}"
echo "limit: ${RP_LIMIT:-30}"
rc=0
"${PYTHON}" scripts/run_report_pages.py \
  --roots "${RP_ROOTS}" --runs-tsv "${RP_RUNS}" --health "${RP_HEALTH}" \
  --downstream "${RP_DOWN}" --out-dir "${RP_OUT}" --index "${RP_INDEX}" \
  --asof "${RP_ASOF}" --limit "${RP_LIMIT:-30}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
