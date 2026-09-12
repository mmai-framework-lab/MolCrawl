#!/bin/bash
#SBATCH --job-name=mc-gen-fold-analysis
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Fold-wise ClinVar analysis: per-fold AUROC for each run, the paired difference
# against a baseline with a variant-level bootstrap interval, the rank test
# between subset groups, and the correlations.
#
# BASELINE_KEY names the field holding the baseline's per-variant score, because
# the two comparisons read different files:
#
#   zero-shot     clinvar_zeroshot_baselines.py --scores-out   key "markov"
#   linear probe  the probe's own model-free control            key "score"
#
# SCORES_DIR must contain only the runs. The probe writes its control beside
# them as model_free/, and that directory would otherwise be read as a run and
# compared against itself.
#
# The GPU is not used; gpu is the only partition here.
#
# Required: SCORES_DIR, BASELINE_SCORES, OUT.
# Optional: BASELINE_KEY, BASELINE_NAME, WINDOW_LOSS, DEGENERATE_BASELINES,
#           PER_RUN_CSV, ROUNDS, SEED.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${SCORES_DIR:?set SCORES_DIR to a directory of <run>/predictions.jsonl}"
: "${BASELINE_SCORES:?set BASELINE_SCORES to the per-variant baseline JSONL}"
: "${OUT:?set OUT to the output json path}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== python:   ${PY} ==="
echo "=== scores:   ${SCORES_DIR} ($(ls -d "${SCORES_DIR}"/*/ 2>/dev/null | wc -l) runs) ==="
echo "=== baseline: ${BASELINE_SCORES} key=${BASELINE_KEY:-markov} ==="
echo "=== out:      ${OUT} ==="

"${PY}" scripts/clinvar_final_analysis.py \
    --scores-dir "${SCORES_DIR}" \
    --baseline-scores "${BASELINE_SCORES}" \
    --baseline-key "${BASELINE_KEY:-markov}" \
    ${BASELINE_NAME:+--baseline-name "${BASELINE_NAME}"} \
    ${WINDOW_LOSS:+--window-loss "${WINDOW_LOSS}"} \
    ${DEGENERATE_BASELINES:+--degenerate-baselines "${DEGENERATE_BASELINES}"} \
    ${PER_RUN_CSV:+--per-run-csv "${PER_RUN_CSV}"} \
    --rounds "${ROUNDS:-2000}" \
    --seed "${SEED:-1026}" \
    --out "${OUT}"
rc=$?
echo ""; echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
