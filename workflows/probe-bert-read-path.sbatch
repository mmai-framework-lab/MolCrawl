#!/bin/bash
#SBATCH --job-name=bert-read-path
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Arrow vs memmap for the BERT read, with no GPU held. The read is separable
# from the step, so it is measured before asking for four GPUs for the arms.
#
#   sbatch --export=ALL,ARROW_DIR=<hf dataset>,BIN_DIR=<memmap dir> \
#          workflows/probe-bert-read-path.sbatch
set -eo pipefail
: "${ARROW_DIR:?set ARROW_DIR}"
: "${BIN_DIR:?set BIN_DIR}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"
else PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== bert read path on ${SLURMD_NODENAME} @ $(date -Is) ==="
rc=0
"${PYTHON}" scripts/probe_bert_read_path.py \
    --arrow-dir "${ARROW_DIR}" --bin-dir "${BIN_DIR}" \
    --procs "${PROCS:-4}" --rows-each "${ROWS_EACH:-25600}" --access "${ACCESS:-both}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
