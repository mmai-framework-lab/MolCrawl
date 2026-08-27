#!/bin/bash
#SBATCH --job-name=mc-mlm-plateau
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Read where a finished MLM run's curve flattens, from its saved checkpoints.
#
# Needed because run length has to be chosen before spending on the production run,
# and the endpoints recorded during those runs came from the diagnostic that dropped
# position_ids. Scoring the checkpoints with the corrected code gives the curve the
# run should have reported. No training, so this is minutes of GPU per point.
#
#   RUN_DIR   the run whose checkpoints are scanned
#   DATA      the split those checkpoints trained on
#   PYTHON    interpreter (a worktree does not contain miniconda)
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
: "${RUN_DIR:?export RUN_DIR}" "${DATA:?export DATA}" "${OUT_DIR:?export OUT_DIR}" "${LABEL:?export LABEL}"
PYTHON="${PYTHON:-${REPO}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}; export PYTHON"; exit 1; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p "${OUT_DIR}" workflows/slurm-logs
echo "=== plateau scan [${LABEL}] ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
rc=0
for ckpt in $(ls -d "${RUN_DIR}"/checkpoint-* | sort -t- -k2 -n); do
  step="${ckpt##*checkpoint-}"
  echo "--- ${LABEL} @ step ${step} ---"
  "${PYTHON}" scripts/eval_mlm_checkpoint.py \
    --checkpoint "${ckpt}" --dataset-dir "${DATA}" --max-length 1024 --document-masking \
    --eval-subset random --label "${LABEL}@${step}" \
    --output "${OUT_DIR}/${LABEL}_step${step}.json" || rc=$?
done
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
