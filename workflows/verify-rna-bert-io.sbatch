#!/bin/bash
#SBATCH --job-name=rna-bert-io-verify
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Count what the RNA BERT run reads. No GPU: it opens Arrow metadata and stats files.
# It still belongs on a compute node -- walking 565 shards on Lustre is not a login-node job.
#
#   sbatch --export=ALL,ARROW_DIR=<hf dataset dir>[,BIN_DIR=<memmap dir>] \
#          workflows/verify-rna-bert-io.sbatch
set -eo pipefail
: "${ARROW_DIR:?set ARROW_DIR to the BERT training_ready_hf_dataset directory}"
[ -d "${ARROW_DIR}" ] || { echo "ARROW_DIR=${ARROW_DIR} is not a directory" >&2; exit 1; }

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"
else PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== rna bert io verify on ${SLURMD_NODENAME} @ $(date -Is) ==="
rc=0
"${PYTHON}" scripts/verify_rna_bert_io.py --arrow-dir "${ARROW_DIR}" \
    ${BIN_DIR:+--bin-dir "${BIN_DIR}"} || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
