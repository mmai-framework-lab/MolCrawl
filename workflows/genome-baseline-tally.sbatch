#!/bin/bash
#SBATCH --job-name=mc-gen-baseline
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Tally one build's degenerate baseline, for a figure that would otherwise have
# to borrow a neighbouring build's number.
#
# The baseline is the loss a model pays by ignoring context and predicting the
# marginal base distribution of the split it is scored on. It therefore belongs
# to a build, not to a subset: two builds of the same subset hold different
# valid rows, and 1.3703 -- carried for the whole campaign -- is one subset's
# value at one window.
#
# MODELS narrows the tally to the side that needs measuring. It reads every
# token of every row, so leaving a measured model in doubles the wall time to
# reproduce a number that already exists.
#
# No GPU work happens here. It runs on the gpu partition because that is where
# compute nodes are, and nothing in this repo runs on the login node.
#
# Required: SRC_ROOT, OUT. Optional: BERT_ROOT (defaults to SRC_ROOT), SUBSETS,
# MODELS (default bert), SPLIT, ROWS.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${SRC_ROOT:?set SRC_ROOT to the genome_sequence root of the build to measure}"
: "${OUT:?set OUT to the output json path}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== build:  ${SRC_ROOT} ==="
echo "=== models: ${MODELS:-bert}   split: ${SPLIT:-valid}   rows: ${ROWS:-all} ==="
echo "=== out:    ${OUT} ==="

# The subset list is read from the environment, never from the --export string,
# where commas separate --export's own assignments and a joined value arrives
# truncated at its first element with no error.
"${PY}" scripts/genome_degenerate_baselines.py \
    --src-root "${SRC_ROOT}" \
    --bert-root "${BERT_ROOT:-${SRC_ROOT}}" \
    --models "${MODELS:-bert}" \
    --split "${SPLIT:-valid}" \
    ${SUBSETS:+--subsets "${SUBSETS}"} \
    ${ROWS:+--rows "${ROWS}"} \
    --out "${OUT}"
rc=$?
echo ""; echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
exit ${rc}
