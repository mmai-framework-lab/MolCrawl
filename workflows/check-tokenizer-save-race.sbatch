#!/bin/bash
#SBATCH --job-name=tokenizer-save-race
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# No GPU. Many concurrent executions of a config that saves and reloads its
# tokenizer, against a private copy of the tree so the shared one is not written.
#
#   sbatch --export=ALL,LEARNING_SOURCE_DIR=<root>,SCRATCH_BASE=<dir>,MODALITY=rna,\
# CONFIG=molcrawl/tasks/pretrain/configs/rna/bert_small.py workflows/check-tokenizer-save-race.sbatch
set -eo pipefail
: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR}"
: "${SCRATCH_BASE:?set SCRATCH_BASE to a directory outside any input tree}"
: "${MODALITY:?set MODALITY}"
: "${CONFIG:?set CONFIG}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"
else PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYROOT}/miniconda/bin/python"
cd "${REPO}"; mkdir -p workflows/slurm-logs

OVERLAY="${SCRATCH_BASE}/race-overlay-${SLURM_JOB_ID:-0}"
rm -rf "${OVERLAY}"; mkdir -p "${OVERLAY}/${MODALITY}"
for e in "${LEARNING_SOURCE_DIR}/${MODALITY}"/*; do
    b="$(basename "${e}")"; case "${b}" in custom_tokenizer_*) continue;; esac
    ln -s "${e}" "${OVERLAY}/${MODALITY}/${b}"
done
echo "=== private tree: ${OVERLAY} (tokenizer directories start absent) ==="
export LEARNING_SOURCE_DIR="${OVERLAY}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1

rc=0
"${PYTHON}" scripts/check_tokenizer_save_race.py --config "${CONFIG}" \
    --procs "${PROCS:-8}" --iters "${ITERS:-6}" 2>&1 | grep -v "FutureWarning\|warnings.warn" || rc=$?
rm -rf "${OVERLAY}"
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
