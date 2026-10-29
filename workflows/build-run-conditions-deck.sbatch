#!/bin/bash
#SBATCH --job-name=mc-cond-deck
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
#
# Directive all-checkpoint-metric-order-2026-10-28-v2 §9: the §7 table as a .pptx,
# with the same numbers in a TSV beside it.
#
# Lays out an existing audit report; measures nothing. The allocation is here
# because this cluster has one partition and nothing heavier than a shell runs on
# the login node.
#
#   sbatch --export=ALL,AUDIT_JSON=<...>,OUT_PPTX=<...>,OUT_TSV=<...> \
#          workflows/build-run-conditions-deck.sbatch
set -eo pipefail

: "${AUDIT_JSON:?set AUDIT_JSON to the checkpoint-metric-match report}"

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

OUT_DIR="${REPORT_DIR:-${PYROOT}/tmp}"
OUT_PPTX="${OUT_PPTX:-${OUT_DIR}/run-conditions-${SLURM_JOB_ID}.pptx}"
OUT_TSV="${OUT_TSV:-${OUT_DIR}/run-conditions-deck-${SLURM_JOB_ID}.tsv}"
mkdir -p "${OUT_DIR}"

echo "=== run conditions deck ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code  : $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "audit : ${AUDIT_JSON}"
echo "pptx  : ${OUT_PPTX}"
echo "tsv   : ${OUT_TSV}"
echo

"${PYTHON}" scripts/build_run_conditions_deck.py \
  --audit "${AUDIT_JSON}" --out "${OUT_PPTX}" --tsv "${OUT_TSV}" \
  ${DECK_ALL:+--all} && rc=0 || rc=$?
# Taken explicitly: under `set -e` a trailing $? can only be 0.
echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
