#!/bin/bash
#SBATCH --job-name=mc-molnl-grid-data-order
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Write the three tables of the 2026-11-26 molnl data order (items 2, 3 and 4).
#
# No GPU: it reads trainer_state.json, nanoGPT's logging CSVs and checkpoint JSONs. It
# still goes through the scheduler -- nothing computes on the login node.
#
#   sbatch --export=ALL,ARMS=<arm inventory tsv>,RUNS_ROOT=<MODEL_OUTPUT_ROOT>,\
# SUMMARY_1125=<tsv>,MANIFESTS_1125=<tsv>,OUT_DIR=<report dir>,STAMP=YYYYMMDD \
#          workflows/molnl-grid-data-order.sbatch
#
# RUNS_ROOT is the PARENT of molecule_nat_lang/, as for molnl-bsweep-report.sbatch.
set -eo pipefail

for v in ARMS RUNS_ROOT SUMMARY_1125 MANIFESTS_1125 OUT_DIR STAMP; do
  [ -n "${!v}" ] || { echo "set ${v}" >&2; exit 1; }
done

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
[ -d "${RUNS_ROOT}/molecule_nat_lang" ] || {
  echo "RUNS_ROOT=${RUNS_ROOT} has no molecule_nat_lang/ under it" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== molnl grid data order on ${SLURMD_NODENAME} @ $(date -Is) commit $(git rev-parse --short HEAD) ==="
"${PYTHON}" scripts/molnl_grid_data_order.py \
  --arms "${ARMS}" --root "${RUNS_ROOT}" \
  --summary-1125 "${SUMMARY_1125}" --manifests-1125 "${MANIFESTS_1125}" \
  --out-dir "${OUT_DIR}" --stamp "${STAMP}"
echo "=== rc=0 @ $(date -Is) ==="
