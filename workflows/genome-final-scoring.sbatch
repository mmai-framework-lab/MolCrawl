#!/bin/bash
#SBATCH --job-name=mc-gen-score
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Score finished genome BERT runs the way the campaign compares them.
#
# The run selected its checkpoint on the first 10,000 valid rows, which is the
# range it evaluated on all the way through. That selection stands. What changes
# here is the number reported: it is measured on the subset's whole valid split,
# under three masking draws, so the comparison across subsets is not resting on
# the first fifth of one split or on a single draw.
#
# The degenerate baseline comes out of the same pass, from the labels at exactly
# the positions that were masked, so "how far above its own degenerate solution"
# is measured at one set of positions rather than two.
#
# Required: RUNS_ROOT, DATA_ROOT, TOKENIZER.
# Optional: SUBSETS, RUN_TAG, SEEDS, ROWS, OUT_DIR, BATCH.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RUNS_ROOT:?set RUNS_ROOT to the directory holding the run outputs}"
: "${DATA_ROOT:?set DATA_ROOT to the genome_sequence root of the scored build}"
: "${TOKENIZER:?set TOKENIZER to the genome single-nucleotide tokenizer}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
TAG="${RUN_TAG:-w1026}"
OUT_DIR="${OUT_DIR:-$PWD/tmp/genome-final-scores}"
mkdir -p "${OUT_DIR}"

# Named subsets are checked against the data being present, the same condition
# the scan below uses -- and the list is read from the environment, never from
# the --export string, where commas separate --export's own assignments and a
# joined value arrives truncated at its first element with no error.
if [ -n "${SUBSETS:-}" ]; then
    LIST="${SUBSETS//,/ }"
    _bad=""
    for s in ${LIST}; do
        [ -d "${RUNS_ROOT}/bert-small-${s}${TAG:+-${TAG}}" ] || _bad="${_bad} ${s}"
    done
    if [ -n "${_bad}" ]; then
        echo "REFUSING: no run directory under RUNS_ROOT for:${_bad}" >&2
        exit 1
    fi
else
    LIST=$(for d in "${RUNS_ROOT}"/bert-small-*"${TAG:+-${TAG}}"/; do
               [ -d "${d}" ] && basename "${d}" | sed "s/^bert-small-//;s/-${TAG}\$//"
           done | sort)
fi

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} subsets=$(echo ${LIST} | wc -w) @ $(date -Is) ==="
rc=0
for s in ${LIST}; do
    echo ""; echo "########## ${s} ##########"
    "${PY}" scripts/score_genome_bert_final.py \
        --run-dir "${RUNS_ROOT}/bert-small-${s}${TAG:+-${TAG}}" \
        --dataset-dir "${DATA_ROOT}/${s}/training_ready_hf_dataset_bert" \
        --tokenizer "${TOKENIZER}" \
        --split valid \
        --seeds "${SEEDS:-1026,1027,1028}" \
        --batch-size "${BATCH:-8}" \
        ${ROWS:+--rows "${ROWS}"} \
        --out "${OUT_DIR}/${s}.json" || { echo "  FAILED: ${s}"; rc=1; }
done
echo ""; echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
