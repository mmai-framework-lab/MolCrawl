#!/bin/bash
#SBATCH --job-name=mc-gen-verifyderived
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Check a derived BERT dataset against the GPT-2 one it came from: row order,
# split inheritance, dtype and declared length. Runs as a job because comparing
# positions across 46M rows is not a login-node operation.
#
# valid and test are compared row for row -- the campaign's reported numbers are
# computed on those rows, so one row of the wrong provenance changes a reported
# value. train is sampled; the script states how in its own output. That is what
# the 12-hour limit is for: the exhaustive splits add 100,000 rows per subset.
#
# The GPU is not used. It is requested because gpu is the only partition here.
#
# Required: SRC_ROOT, DST_ROOT. Optional: SUBSETS, SAMPLE (overrides the
# script's own stride count; leave unset to take its default).

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${SRC_ROOT:?}"; : "${DST_ROOT:?}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# A named list bypasses the directory scan below, which is what filters out
# entries that are not subsets at all -- DST_ROOT also holds the tokenizer
# directory. Names are therefore checked against the same condition the scan
# uses, up front, so a typo or a stray entry fails here rather than three
# subsets into a job.
#
# The list is read from the environment, never from the --export string: commas
# separate --export's own assignments, so a comma-joined value there arrives
# truncated at the first element with no error.
if [ -n "${SUBSETS:-}" ]; then
    LIST="${SUBSETS//,/ }"
    _bad=""
    for s in ${LIST}; do
        [ -d "${DST_ROOT}/${s}/training_ready_hf_dataset_bert" ] || _bad="${_bad} ${s}"
    done
    if [ -n "${_bad}" ]; then
        echo "REFUSING: no training_ready_hf_dataset_bert under DST_ROOT for:${_bad}" >&2
        echo "          DST_ROOT=${DST_ROOT}" >&2
        exit 1
    fi
    echo "=== SUBSETS named: $(echo ${LIST} | wc -w) ==="
else LIST=$(for d in "${DST_ROOT}"/*/; do
                [ -d "${d}/training_ready_hf_dataset_bert" ] && basename "${d}"; done | sort); fi

echo "=== node=$(hostname) subsets=$(echo ${LIST} | wc -w) ==="
rc=0
for s in ${LIST}; do
    echo ""; echo "########## ${s} ##########"
    "${PY}" scripts/verify_bert_from_gpt2.py \
        "${SRC_ROOT}/${s}/training_ready_hf_dataset_gpt2" \
        "${DST_ROOT}/${s}/training_ready_hf_dataset_bert" \
        ${SAMPLE:+--sample "${SAMPLE}"} || { echo "  FAILED: ${s}"; rc=1; }
done
echo ""; echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
