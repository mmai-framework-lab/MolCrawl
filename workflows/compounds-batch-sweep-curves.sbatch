#!/bin/bash
#SBATCH --job-name=mc-cmp-curves
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Read the compounds batch-size grid's evaluation series out of the run logs and write
# the curve TSV (section 12) and the per-arm summary (sections 10 and 5).
#
# No GPU: it reads text. It runs here rather than on the login node because compute on
# the login node is not allowed here.
#
#   sbatch --export=ALL,ARMS=<...>/rt1-...tsv,LOGS=<log dir>,CURVES_OUT=<...>/rt1-...tsv,SUMMARY_OUT=<...>/rt1-...tsv \
#          workflows/compounds-batch-sweep-curves.sbatch
#
# ARCH defaults to gpt2, which reads the run logs and needs LOGS. ARCH=bert reads each
# run's newest checkpoint and needs RUNS, the directory holding the run output
# directories. A value passed through --export must not contain a comma: sbatch splits
# the list on commas and the value is truncated there.
set -uo pipefail

: "${ARMS:?set ARMS to the derived arm inventory TSV}"
: "${CURVES_OUT:?set CURVES_OUT to the curve TSV to write}"
: "${SUMMARY_OUT:?set SUMMARY_OUT to the per-arm summary TSV to write}"
ARCH="${ARCH:-gpt2}"
if [ "${ARCH}" = "bert" ]; then
  : "${RUNS:?ARCH=bert reads trainer_state.json; set RUNS to the run output root}"
  READ_FROM=(--runs "${RUNS}")
else
  : "${LOGS:?ARCH=gpt2 reads the run logs; set LOGS to the log directory}"
  READ_FROM=(--logs "${LOGS}")
fi

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

echo "=== compounds batch-sweep curves on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "arch: ${ARCH}"
echo "read from: ${READ_FROM[*]}"

echo "########## 1. the curves and the per-arm summary ##########"
"${PYTHON}" scripts/compounds_batch_sweep_curves.py \
    --arms "${ARMS}" "${READ_FROM[@]}" --arch "${ARCH}" \
    --curves-out "${CURVES_OUT}" --summary-out "${SUMMARY_OUT}"
rc=$?
echo "-- curves rc=${rc}"

echo
echo "########## 2. unit tests ##########"
"${PYTHON}" -m pytest -q tests/unit/test_compounds_batch_sweep_curves.py
trc=$?
echo "-- pytest rc=${trc}"

echo
echo "=== rc=$((rc + trc)) @ $(date -Is) ==="
exit $((rc + trc))
