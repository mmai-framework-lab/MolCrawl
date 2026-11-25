#!/bin/bash
#SBATCH --job-name=mc-cmp-report-check
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Check a compounds progress report against the TSVs it quotes, before it is sent.
#
# No GPU: it reads two TSVs and a markdown file. It runs here rather than on the
# login node because compute on the login node is not allowed here.
#
#   sbatch --export=ALL,REPORT=<...>.md,SUMMARY=<...>.tsv,CURVES=<...>.tsv \
#          workflows/compounds-progress-report-check.sbatch
#
# A value passed through --export must not contain a comma: sbatch splits the
# list on commas and the value is truncated there.
set -uo pipefail

: "${REPORT:?set REPORT to the progress report to check}"
: "${SUMMARY:?set SUMMARY to the arm summary TSV the report quotes}"
: "${CURVES:?set CURVES to the curve TSV the report quotes}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== compounds progress report check on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "report: ${REPORT}"

"${PYTHON}" scripts/check_compounds_progress_report.py \
    --report "${REPORT}" --summary "${SUMMARY}" --curves "${CURVES}"
rc=$?

echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
