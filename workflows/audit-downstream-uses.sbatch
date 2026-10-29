#!/bin/bash
#SBATCH --job-name=mc-downstream-uses
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Directive all-checkpoint-metric-order-2026-10-28-v2 §5: which downstream
# evaluations used the runs §4 flagged. Reads result files, re-runs nothing.
#
# AUDIT_JSON is the report written by workflows/audit-checkpoint-metric-match.sbatch.
# SEARCH_DIRS is a whitespace-separated list of directories holding downstream
# results. Both required and unset-fatal, because they are gitignored paths that
# differ per checkout.
#
#   sbatch --export=ALL,AUDIT_JSON=<...>,SEARCH_DIRS="<dir> <dir>" \
#          workflows/audit-downstream-uses.sbatch
set -eo pipefail

: "${AUDIT_JSON:?set AUDIT_JSON to the checkpoint-metric-match report}"
: "${SEARCH_DIRS:?set SEARCH_DIRS to a whitespace-separated list of result directories}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO}"; mkdir -p workflows/slurm-logs
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

[ -f "${AUDIT_JSON}" ] || { echo "no such file: ${AUDIT_JSON}" >&2; exit 1; }

SEARCH_ARGS=()
for dir in ${SEARCH_DIRS}; do
  [ -d "${dir}" ] || { echo "not a directory: ${dir}" >&2; exit 1; }
  SEARCH_ARGS+=(--search "${dir}")
done

OUT_DIR="${REPORT_DIR:-${PYROOT}/tmp}"
REPORT="${REPORT:-${OUT_DIR}/downstream-uses-${SLURM_JOB_ID}.json}"
mkdir -p "${OUT_DIR}"

echo "=== downstream uses ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code  : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "audit : ${AUDIT_JSON}"
echo "search: ${SEARCH_DIRS}"
echo "report: ${REPORT}"
echo

"${PYTHON}" scripts/audit_downstream_uses_of_runs.py \
  --audit "${AUDIT_JSON}" --match "${MATCH:-no}" \
  "${SEARCH_ARGS[@]}" --report "${REPORT}" && rc=0 || rc=$?
# Taken explicitly: under `set -e` a trailing $? can only be 0.
echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
