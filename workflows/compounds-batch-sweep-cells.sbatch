#!/bin/bash
#SBATCH --job-name=mc-cmp-cells
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Per-cell and per-arm tables for the compounds batch-size grid: where the winning
# rate sat among the rates its cell swept, and which value represents each arm.
#
# No GPU: it reads TSVs. It runs here rather than on the login node because compute
# on the login node is not allowed here.
#
#   sbatch --export=ALL,ARMS=<...>.tsv,SUMMARY=<...>.tsv,CURVES=<...>.tsv,CELLS_OUT=<...>.tsv,ARM_TABLE_OUT=<...>.tsv \
#          workflows/compounds-batch-sweep-cells.sbatch
#
# SPREAD is optional; when set, the best batch at each fraction is printed too.
# A value passed through --export must not contain a comma: sbatch splits the list
# on commas and the value is truncated there.
set -uo pipefail

: "${ARMS:?set ARMS to the derived arm inventory TSV}"
: "${SUMMARY:?set SUMMARY to the per-arm summary TSV}"
: "${CURVES:?set CURVES to the curve TSV}"
: "${CELLS_OUT:?set CELLS_OUT to the per-cell TSV to write}"
: "${ARM_TABLE_OUT:?set ARM_TABLE_OUT to the per-arm TSV to write}"

# A pass queued behind a run does not know what day it will be when it fires, and a
# report file is named for the day its contents are from. The literal string DATE in
# an output path is replaced with the day the pass actually ran.
_today="$(date +%Y-%m-%d)"
CELLS_OUT="${CELLS_OUT//DATE/${_today}}"
ARM_TABLE_OUT="${ARM_TABLE_OUT//DATE/${_today}}"

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

echo "=== compounds batch-sweep cells on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
for f in "${ARMS}" "${SUMMARY}" "${CURVES}"; do
  [ -s "${f}" ] || { echo "missing input: ${f}" >&2; exit 1; }
done

echo "########## 1. unit tests ##########"
"${PYTHON}" -m pytest -q tests/unit/test_compounds_batch_sweep_cells.py
trc=$?
echo "-- pytest rc=${trc}"

echo
echo "########## 2. the tables ##########"
CELL_ARGS=(--arms "${ARMS}" --summary "${SUMMARY}" --curves "${CURVES}"
           --cells-out "${CELLS_OUT}" --arm-table-out "${ARM_TABLE_OUT}")
[ -n "${SPREAD:-}" ] && CELL_ARGS+=(--spread "${SPREAD}")
"${PYTHON}" scripts/compounds_batch_sweep_cells.py "${CELL_ARGS[@]}"
rc=$?
echo "-- cells rc=${rc}"

echo
echo "=== rc=$((rc + trc)) @ $(date -Is) ==="
exit $((rc + trc))
