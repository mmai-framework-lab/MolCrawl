#!/bin/bash
#SBATCH --job-name=mc-gen-sched-fig
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# The schedule measurement's figures, drawn from the loss-curve tables.
#
# Two tables in, two figures and one table out. Nothing is read from a run
# directory here: the curves come from the same TSV every other genome figure is
# drawn from, so a value on a chart can always be found in it.
#
# RUN THE CURVE JOB FIRST if the measurement has advanced since the table was
# built. This job fails rather than drawing a partial curve when the subset is
# missing from a series, but it cannot tell a finished run from one the table
# caught halfway.
#
# The GPU is not used. gpu is the only partition with compute nodes here.
#
# Required: CURVES_DIR (holding genome-loss-curves.tsv and genome-loss-per-run.tsv).
# Optional: OUT_DIR, SUBSET, BASELINE, PER_STEP, TAIL_BELOW.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${CURVES_DIR:?set CURVES_DIR to the directory holding the loss-curve tables}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
OUT_DIR="${OUT_DIR:-${CURVES_DIR}}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== tables: ${CURVES_DIR} ==="
echo "=== out:    ${OUT_DIR} ==="

"${PY}" scripts/plot_genome_schedule_measurement.py \
    --tsv "${CURVES_DIR}/genome-loss-curves.tsv" \
    --per-run "${CURVES_DIR}/genome-loss-per-run.tsv" \
    --subset "${SUBSET:-mammal_centered}" \
    ${BASELINE:+--baseline "${BASELINE}"} \
    --per-step "${PER_STEP:-2621440}" \
    --tail-below "${TAIL_BELOW:-1.12}" \
    --out-dir "${OUT_DIR}"
rc=$?
echo ""; echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
exit ${rc}
