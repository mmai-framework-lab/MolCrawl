#!/bin/bash
#SBATCH --job-name=mc-gen-curves
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Build the genome loss-curve table, then draw every figure from it.
#
# The table comes first and the figures are drawn from nothing else, so a value
# on a chart can always be found in the TSV and the two cannot disagree.
#
# The five series are kept apart at every step. Their windows are 512, 1,024
# and 1,026 and their lengths are 3 and 9 epochs, so a curve from one says
# nothing about a curve from another; they share only the metric, which is
# pinned to eval_loss_mask for BERT rather than taken from whichever metric
# each run selected on.
#
# A baseline is drawn only where one was measured on that series' own build.
# The 1,026-window BERT runs and the GPT-2 runs have one (2026-09-16, valid all
# 50,000 rows); the 512-window series and the 1,024 probe do not, and the
# figures say so rather than borrowing a neighbouring build's number.
#
# No GPU work happens here. It runs on the gpu partition because that is where
# compute nodes are, and nothing in this repo runs on the login node.
#
# Required: RUNS_ROOT, OUT_DIR. Optional: RESULTS_ROOT, SACCT_START, SACCT_END.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RUNS_ROOT:?set RUNS_ROOT to the directory holding the run outputs}"
: "${OUT_DIR:?set OUT_DIR to the report directory the figures and TSV go to}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
RESULTS_ROOT="${RESULTS_ROOT:-${RUNS_ROOT}/_results}"
mkdir -p "${OUT_DIR}"
TSV="${OUT_DIR}/genome-loss-curves.tsv"
META="${OUT_DIR}/genome-loss-curves-meta.json"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== python:  ${PY} ==="
echo "=== runs:    ${RUNS_ROOT} ==="
echo "=== outputs: ${OUT_DIR} ==="

echo ""; echo "########## 1. TSV ##########"
"${PY}" scripts/genome_loss_curves.py \
    --runs-root "${RUNS_ROOT}" \
    --log-dir "$PWD/workflows/slurm-logs" \
    --sacct-start "${SACCT_START:-2026-08-16}" \
    --sacct-end "${SACCT_END:-2026-08-20}" \
    --out "${TSV}" \
    --meta-out "${META}" || { echo "  FAILED: TSV"; exit 1; }

echo ""; echo "########## 2. 図 ##########"
# series=directory. A series absent from this list gets no baseline line, which
# is the honest state for the builds that have not been measured.
"${PY}" scripts/plot_genome_loss_curves.py \
    --tsv "${TSV}" \
    --meta "${META}" \
    --baselines \
        "bert-w1026=${RESULTS_ROOT}/final-scores-bert-w1026" \
        "gpt2=${RESULTS_ROOT}/final-scores-gpt2" \
    --out-dir "${OUT_DIR}" || { echo "  FAILED: figures"; exit 1; }

echo ""; echo "---- rc=0 ----"
ls -la "${OUT_DIR}" | grep -E "genome-loss" | sed 's/^/  /'
echo "=== done @ $(date -Is) ==="
