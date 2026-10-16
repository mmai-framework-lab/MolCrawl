#!/bin/bash
#SBATCH --job-name=mc-audit-data
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Measure each modality's splits and row length from the arrow, so a recorded figure
# can be checked against the data rather than against dataset_info.json.
#
#   sbatch --export=ALL,AUDIT_REPORT=<json>,AUDIT_DATASETS="name=path name=path" \
#          workflows/audit-data-parameters.sbatch
set -uo pipefail
: "${AUDIT_REPORT:?export AUDIT_REPORT to the JSON to write}"
: "${AUDIT_DATASETS:?export AUDIT_DATASETS as a space-separated list of name=path}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== data parameter audit on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
ARGS=""
for d in ${AUDIT_DATASETS}; do ARGS="${ARGS} --dataset ${d}"; done
# shellcheck disable=SC2086 - ARGS is built as separate words on purpose
"${PYTHON}" scripts/audit_data_parameters.py ${ARGS} --report "${AUDIT_REPORT}"
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
