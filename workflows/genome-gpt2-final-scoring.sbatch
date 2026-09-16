#!/bin/bash
#SBATCH --job-name=mc-gen-score-gpt2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Score finished genome GPT-2 runs on the range the campaign compares them on.
#
# The number carried so far, best_val, is the smallest value the trainer's own
# evaluation produced: 320 x 200 = 64,000 rows drawn with replacement from the
# 50,000-row valid split, under bf16 autocast, minimised over every evaluation
# point. The degenerate baseline it was subtracted from came from a different
# range again -- the first 10,000 rows. A margin between two such terms carries
# the range difference inside it.
#
# This scores the adopted checkpoint over every valid row once, in order, and
# takes the baseline from the labels at exactly the positions that paid loss.
# It is the GPT-2 counterpart of genome-final-scoring.sbatch; run both before
# reporting the second correlation.
#
# Required: RUNS_ROOT, DATA_ROOT, TOKENIZER.
# Optional: SUBSETS, RUN_TAG, ROWS, OUT_DIR, BATCH.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RUNS_ROOT:?set RUNS_ROOT to the directory holding the run outputs}"
: "${DATA_ROOT:?set DATA_ROOT to the genome_sequence root of the scored build}"
: "${TOKENIZER:?set TOKENIZER to the genome single-nucleotide tokenizer}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# The GPT-2 runs carry no window tag in their directory names, so RUN_TAG
# defaults to empty. ${RUN_TAG-} and not ${RUN_TAG:-}: an explicitly empty
# RUN_TAG must stay empty rather than fall back to a default.
TAG="${RUN_TAG-}"
OUT_DIR="${OUT_DIR:-${RUNS_ROOT}/_results/final-scores-gpt2}"
mkdir -p "${OUT_DIR}"

# The subset list is read from the environment, never from the --export string,
# where commas separate --export's own assignments and a joined value arrives
# truncated at its first element with no error. Named subsets are checked
# against the run directory being present, the same condition the scan uses.
if [ -n "${SUBSETS:-}" ]; then
    LIST="${SUBSETS//,/ }"
    _bad=""
    for s in ${LIST}; do
        [ -f "${RUNS_ROOT}/gpt2-small-${s}${TAG:+-${TAG}}/ckpt.pt" ] || _bad="${_bad} ${s}"
    done
    if [ -n "${_bad}" ]; then
        echo "REFUSING: no adopted ckpt.pt under RUNS_ROOT for:${_bad}" >&2
        exit 1
    fi
else
    LIST=$(for d in "${RUNS_ROOT}"/gpt2-small-*"${TAG:+-${TAG}}"/; do
               [ -f "${d}/ckpt.pt" ] && basename "${d}" | sed "s/^gpt2-small-//;s/-${TAG}\$//"
           done | sort)
fi

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} subsets=$(echo ${LIST} | wc -w) @ $(date -Is) ==="
echo "=== python:  ${PY} ==="
echo "=== runs:    ${RUNS_ROOT} ==="
echo "=== outputs: ${OUT_DIR} ==="
rc=0
for s in ${LIST}; do
    echo ""; echo "########## ${s} ##########"
    "${PY}" scripts/score_genome_gpt2_final.py \
        --run-dir "${RUNS_ROOT}/gpt2-small-${s}${TAG:+-${TAG}}" \
        --dataset-dir "${DATA_ROOT}/${s}/training_ready_hf_dataset_gpt2" \
        --tokenizer "${TOKENIZER}" \
        --split valid \
        --batch-size "${BATCH:-32}" \
        ${ROWS:+--rows "${ROWS}"} \
        --out "${OUT_DIR}/${s}.json" || { echo "  FAILED: ${s}"; rc=1; }
done
echo ""; echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
