#!/bin/bash
# scib-metrics が実データで動くかを確かめる。評価専用の環境で走らせる。
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --job-name=rna-scib-smoke
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${EVAL_ENV_DIR:?export EVAL_ENV_DIR to the evaluation venv (see rna-eval-env.sbatch)}"
: "${EVAL_H5AD:?export EVAL_H5AD to the evaluation AnnData}"
ENV_PY="${EVAL_ENV_DIR}/bin/python"
[ -x "${ENV_PY}" ] || { echo "no python under ${EVAL_ENV_DIR}" >&2; exit 1; }
export PYTHONUNBUFFERED=1
mkdir -p workflows/slurm-logs
echo "=== rna-scib-smoke @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
"${ENV_PY}" scripts/rna_scib_smoke.py --h5ad "${EVAL_H5AD}" --cells "${CELLS:-4000}"
