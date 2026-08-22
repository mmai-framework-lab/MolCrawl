#!/bin/bash
#SBATCH --job-name=mc-mlm-baselines
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Recompute the MLM judgement baselines from the data. No model involved; --gpus=1
# only because `gpu` is the sole partition here.
#
#   PACKED_BERT / ONEMOL  the two data forms the arms used
#   PYTHON                interpreter (a worktree does not contain miniconda)
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
: "${PACKED_BERT:?export PACKED_BERT}" "${ONEMOL:?export ONEMOL}" "${OUT_DIR:?export OUT_DIR}"
PYTHON="${PYTHON:-${REPO}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}; export PYTHON"; exit 1; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p "${OUT_DIR}" workflows/slurm-logs
echo "=== MLM baselines ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
rc=0
"${PYTHON}" scripts/mlm_lookup_baselines.py --dataset-dir "${PACKED_BERT}" --max-length 1024 \
  --label "packed 1024 (W 系が見たもの)" --output "${OUT_DIR}/baselines_packed.json" || rc=$?
"${PYTHON}" scripts/mlm_lookup_baselines.py --dataset-dir "${ONEMOL}" --max-length 128 \
  --label "1 分子 1 行 seq128 (V1L が見たもの)" --output "${OUT_DIR}/baselines_1mol.json" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
