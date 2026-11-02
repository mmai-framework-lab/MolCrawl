#!/bin/bash
#SBATCH --job-name=mc-mlm-baselines-by-config
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# The model-free MLM baselines for one or more modalities.
#
# No GPU: every number here is a count table fitted on train rows and scored at
# the positions the collator masks. It is still not login-node work -- it reads
# the packed datasets and the fit is over hundreds of thousands of rows.
#
# The tokenizer, dataset_dir and masking rate come from each pretrain config, so
# the baseline is fitted and scored where the run was.
#
# Each entry is label|config|LEARNING_SOURCE_DIR|extra env, the last field a
# comma-separated list of NAME=VALUE. Every modality resolves its tokenizer and
# its dataset under its own learning_source root, and genome needs the subset
# named as well, so one shared environment cannot drive all four.
#
#   sbatch --export=ALL,BASELINE_OUT_DIR=<dir>,\
#     BASELINE_CONFIGS="rna|configs/rna/bert_medium_lr3e4.py|/path/to/rna_train| ..." \
#     workflows/mlm-baselines.sbatch
set -eo pipefail
: "${BASELINE_OUT_DIR:?export BASELINE_OUT_DIR to where the JSON results go, outside every input tree}"
: "${BASELINE_CONFIGS:?export BASELINE_CONFIGS as a space-separated list of label|config|source|extra}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
    PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${REPO}"; mkdir -p workflows/slurm-logs "${BASELINE_OUT_DIR}"

echo "=== mlm baselines on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Is) ==="
echo "tree:    ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "outputs: ${BASELINE_OUT_DIR}"
rc=0
for ENTRY in ${BASELINE_CONFIGS}; do
    IFS='|' read -r LABEL CONFIG SOURCE EXTRA <<< "${ENTRY}"
    echo "--- ${LABEL}: ${CONFIG}"
    echo "    LEARNING_SOURCE_DIR=${SOURCE}  extra=${EXTRA:-none}  @ $(date -Is)"
    (
        export LEARNING_SOURCE_DIR="${SOURCE}"
        if [ -n "${EXTRA}" ]; then
            IFS=',' read -ra KVS <<< "${EXTRA}"
            for KV in "${KVS[@]}"; do export "${KV?}"; done
        fi
        "${PYTHON}" scripts/mlm_lookup_baselines.py \
            --config "${CONFIG}" --label "${LABEL}" \
            ${BASELINE_FIT_ROWS:+--fit-rows "${BASELINE_FIT_ROWS}"} \
            ${BASELINE_UNIGRAM_ONLY:+--unigram-only} \
            --output "${BASELINE_OUT_DIR}/${LABEL}.json"
    ) || { rc=$?; echo "!!! ${LABEL} failed rc=${rc}"; }
done
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
