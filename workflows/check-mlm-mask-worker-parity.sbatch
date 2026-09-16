#!/bin/bash
#SBATCH --job-name=mlm-mask-parity
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# No GPU. Compares the masked positions a dataloader produces with 0 workers
# against 4, on identical rows in identical order.
#
#   sbatch --export=ALL,LEARNING_SOURCE_DIR=<root>,TPUT_OUT_BASE=<scratch> \
#          workflows/check-mlm-mask-worker-parity.sbatch
set -eo pipefail
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR}"
: "${TPUT_OUT_BASE:?set TPUT_OUT_BASE}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"
else PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYROOT}/miniconda/bin/python"
cd "${REPO}"; mkdir -p workflows/slurm-logs

# Own tokenizer directory, as the arm harness does: the config writes it on import.
OVERLAY="${TPUT_OUT_BASE}/ls-overlay-mask-${SLURM_JOB_ID:-0}"
rm -rf "${OVERLAY}"; mkdir -p "${OVERLAY}/rna"
for e in "${LEARNING_SOURCE_DIR}"/rna/*; do
    b="$(basename "${e}")"; [ "${b}" = "custom_tokenizer_bert" ] && continue
    ln -s "${e}" "${OVERLAY}/rna/${b}"
done
mkdir -p "${OVERLAY}/rna/custom_tokenizer_bert"
export LEARNING_SOURCE_DIR="${OVERLAY}"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${TPUT_OUT_BASE}/hf-cache-rna"

echo "=== mlm mask parity on ${SLURMD_NODENAME} @ $(date -Is) ==="
rc=0
"${PYTHON}" scripts/check_mlm_mask_worker_parity.py \
    --config molcrawl/tasks/pretrain/configs/rna/bert_small.py \
    --rows "${ROWS:-256}" --workers "${WORKERS:-4}" || rc=$?
rm -rf "${OVERLAY}"
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
