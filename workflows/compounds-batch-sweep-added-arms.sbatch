#!/bin/bash
#SBATCH --job-name=mc-cmp-added
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Derive the arms the 2026-11-27b directive adds, write their configs, and prove the
# 48 running arms' configs did not move.
#
# No GPU: it reads TSVs and writes text. It runs here rather than on the login node
# because compute on the login node is not allowed here.
#
#   sbatch --export=ALL,CELLS=<...>.tsv,ARMS=<...>.tsv,ADDED_OUT=<...>.tsv \
#          workflows/compounds-batch-sweep-added-arms.sbatch
#
# WRITE_CONFIGS=0 stops before the configs are written, so the derivation and the
# checks can be run on their own. A value passed through --export must not contain a
# comma: sbatch splits the list on commas and the value is truncated there.
set -uo pipefail

: "${CELLS:?set CELLS to the measured per-cell TSV}"
: "${ARMS:?set ARMS to the 68-arm inventory TSV}"
: "${ADDED_OUT:?set ADDED_OUT to the added-arm inventory TSV to write}"

# A pass queued behind a run does not know what day it will be when it fires, and a
# report file is named for the day its contents are from. The literal string DATE in an
# output path is replaced with the day the pass actually ran.
_today="$(date +%Y-%m-%d)"
ADDED_OUT="${ADDED_OUT//DATE/${_today}}"

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
CONFIG_DIR="molcrawl/tasks/pretrain/configs/compounds"

echo "=== compounds added arms on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
for f in "${CELLS}" "${ARMS}"; do
  [ -s "${f}" ] || { echo "missing input: ${f}" >&2; exit 1; }
done

echo "########## 1. unit tests ##########"
"${PYTHON}" -m pytest -q tests/unit/test_compounds_batch_sweep_added_arms.py \
                         tests/unit/test_compounds_batch_sweep_cells.py
trc=$?
echo "-- pytest rc=${trc}"

echo
echo "########## 2. the added arms ##########"
"${PYTHON}" scripts/compounds_batch_sweep_added_arms.py \
    --cells "${CELLS}" --arms "${ARMS}" --out "${ADDED_OUT}"
arc=$?
echo "-- added arms rc=${arc}"

wrc=0
crc=0
grc=0
if [ "${WRITE_CONFIGS:-1}" = "1" ] && [ "${arc}" = "0" ]; then
  echo
  echo "########## 3. the 42 configs ##########"
  # The 48 running arms are not in this inventory, so nothing of theirs is rewritten.
  "${PYTHON}" scripts/compounds_batch_sweep_configs.py --arms "${ADDED_OUT}"
  wrc=$?
  echo "-- write + check rc=${wrc}"

  echo
  echo "########## 4. the 68 running arms, unchanged ##########"
  "${PYTHON}" scripts/compounds_batch_sweep_configs.py --arms "${ARMS}" --check
  crc=$?
  echo "-- running arms check rc=${crc}"
  echo "-- git sees in ${CONFIG_DIR}:"
  git status --porcelain "${CONFIG_DIR}" | sed 's/^/     /'
  changed="$(git status --porcelain "${CONFIG_DIR}" | grep -cv '^??' || true)"
  added="$(git status --porcelain "${CONFIG_DIR}" | grep -c '^??' || true)"
  echo "-- tracked configs changed: ${changed} (must be 0); new files: ${added}"
  [ "${changed}" = "0" ] || grc=1
else
  echo
  echo "(configs not written: WRITE_CONFIGS=${WRITE_CONFIGS:-1}, added arms rc=${arc})"
fi

echo
echo "=== rc=$((trc + arc + wrc + crc + grc)) @ $(date -Is) ==="
exit $((trc + arc + wrc + crc + grc))
