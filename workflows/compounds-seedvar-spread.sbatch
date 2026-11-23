#!/bin/bash
#SBATCH --job-name=mc-cmp-seedvar-spread
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Read the three compounds GPT-2 large runs that differ only in the seed and report the
# spread of the end-of-run rise, beside the reference runs (the 2026-11-20 directive's
# section 4).
#
# No GPU: it reads text. It runs here rather than on the login node because compute on
# the login node is not allowed here.
#
#   sbatch --export=ALL,RUNS_ROOT=<runs root>,OUT=<...>/rt1-...tsv \
#          workflows/compounds-seedvar-spread.sbatch
#
# A value passed through --export must not contain a comma: sbatch splits the list on
# commas and the value is truncated there.
set -uo pipefail

: "${RUNS_ROOT:?set RUNS_ROOT to the runs root the arms wrote into}"
: "${OUT:?set OUT to the TSV to write}"

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

G="${RUNS_ROOT}/compounds/gpt2-output"
echo "=== compounds seed-variance spread on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"

rc=0
echo
echo "########## 1. the spread, with the two reference runs ##########"
# The references: the run the +0.5260 came from (2,560, 2026-10-01 code) and the grid
# arm at 2,048 the comparison is against. Neither is part of the spread.
"${PYTHON}" scripts/compounds_seedvar_spread.py \
    --run "seed42=${G}/compounds-large-lr1p6e3-sv42" \
    --run "seed1=${G}/compounds-large-lr1p6e3-seed1" \
    --run "seed17=${G}/compounds-large-lr1p6e3-seed17" \
    --reference "existing-2560=${G}/compounds-large-lr1p6e3" \
    --reference "grid-2048=${G}/compounds-large-b2048-lr1p43e3" \
    --out "${OUT}"
step=$?; echo "-- spread rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"

echo
echo "########## 2. unit tests ##########"
"${PYTHON}" -m pytest -q tests/unit/test_compounds_seedvar_spread.py
step=$?; echo "-- pytest rc=${step}"; [ "${step}" -eq 0 ] || rc="${step}"

echo
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
