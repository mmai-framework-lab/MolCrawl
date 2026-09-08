#!/bin/bash
#SBATCH --job-name=mc-gen-clinvarcount
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Count the ClinVar variants available on the chromosomes every pretraining
# subset held out (chr21 / 22 / X / Y), which is where the upstream evaluation
# is scored.
#
# Restricting it there is what keeps mammal_centered comparable: that subset
# alone carries the human genome and learned chr1-chr20, so a genome-wide
# evaluation would score one run of twenty-one on sequence it had already seen.
# The price is sample size, and the binding figure is the smaller of the two
# classes, since a balanced comparison cannot exceed it.
#
# The GPU is not used; gpu is the only partition here.
#
# Required: CLINVAR_DIR. Optional: CHROMS, OUT_DIR.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${CLINVAR_DIR:?set CLINVAR_DIR to the directory holding clinvar_sequences.csv}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

RESULTS_ROOT="${RESULTS_ROOT:-${PYROOT}/learning_source_genome_runs/_results}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/analyses}"
mkdir -p "${OUT_DIR}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== python:  ${PY} ==="
echo "=== clinvar: ${CLINVAR_DIR} ==="
echo "=== outputs: ${OUT_DIR} ==="

"${PY}" scripts/clinvar_heldout_counts.py \
    --sequences "${CLINVAR_DIR}/clinvar_sequences.csv" \
    --splits "${CLINVAR_DIR}/splits_manifest.csv" \
    --chroms "${CHROMS:-21,22,X,Y}" \
    --out "${OUT_DIR}/clinvar-heldout-counts.json"
rc=$?
echo ""; echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
