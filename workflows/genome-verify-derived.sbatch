#!/bin/bash
#SBATCH --job-name=mc-gen-verifyderived
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Check a derived BERT dataset against the GPT-2 one it came from: row order,
# split inheritance, dtype and declared length. Runs as a job because comparing
# positions across 46M rows is not a login-node operation.
#
# Required: SRC_ROOT, DST_ROOT. Optional: SUBSETS, SAMPLE.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${SRC_ROOT:?}"; : "${DST_ROOT:?}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [ -n "${SUBSETS:-}" ]; then LIST="${SUBSETS//,/ }"
else LIST=$(for d in "${DST_ROOT}"/*/; do
                [ -d "${d}/training_ready_hf_dataset_bert" ] && basename "${d}"; done | sort); fi

echo "=== node=$(hostname) subsets=$(echo ${LIST} | wc -w) ==="
rc=0
for s in ${LIST}; do
    echo ""; echo "########## ${s} ##########"
    "${PY}" scripts/verify_bert_from_gpt2.py \
        "${SRC_ROOT}/${s}/training_ready_hf_dataset_gpt2" \
        "${DST_ROOT}/${s}/training_ready_hf_dataset_bert" \
        --sample "${SAMPLE:-20000}" || { echo "  FAILED: ${s}"; rc=1; }
done
echo ""; echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
