#!/bin/bash
#SBATCH --job-name=mc-cmp-downstream
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Downstream evaluation of the compounds models on frozen representations.
#
#   sbatch --export=ALL,DS_TASK=solubility,DS_DATA=<tab>,DS_REPORT=<json>[,DS_ARGS=...] \
#          workflows/compounds-downstream-eval.sbatch
#
# No network. --gpus is set per submission: the control path needs none.
set -uo pipefail
: "${DS_TASK:?export DS_TASK as solubility|herg}"
: "${DS_DATA:?export DS_DATA to the TDC .tab file}"
: "${DS_REPORT:?export DS_REPORT to the JSON to write}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}"; mkdir -p workflows/slurm-logs "$(dirname "${DS_REPORT}")"
echo "=== compounds downstream (${DS_TASK}) on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
# shellcheck disable=SC2086 - DS_ARGS is meant to split into arguments
"${PYTHON}" scripts/compounds_downstream_eval.py \
    --task "${DS_TASK}" --data "${DS_DATA}" --report "${DS_REPORT}" ${DS_ARGS:-}
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
