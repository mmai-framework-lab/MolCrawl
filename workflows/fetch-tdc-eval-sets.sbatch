#!/bin/bash
#SBATCH --job-name=mc-tdc-fetch
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Pull the downstream evaluation sets while the machine can still reach the network.
# Runs in the TDC venv, not the environment the training runs use.
#
#   sbatch --export=ALL,TDC_ENV_DIR=<venv>,TDC_DATA_DIR=<dir>,TDC_REPORT=<json>,\
#   NETRUN=<wrapper> workflows/fetch-tdc-eval-sets.sbatch
set -uo pipefail
: "${TDC_ENV_DIR:?export TDC_ENV_DIR to the venv built by workflows/tdc-env.sbatch}"
: "${TDC_DATA_DIR:?export TDC_DATA_DIR to the directory TDC should cache into}"
: "${TDC_REPORT:?export TDC_REPORT to the JSON to write}"
NETRUN="${NETRUN:?export NETRUN to the DNS wrapper}"
[ -x "${NETRUN}" ] || { echo "no wrapper at ${NETRUN}" >&2; exit 1; }
ENV_PY="${TDC_ENV_DIR}/bin/python"
[ -x "${ENV_PY}" ] || { echo "no venv at ${TDC_ENV_DIR}" >&2; exit 1; }
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs "${TDC_DATA_DIR}" "$(dirname "${TDC_REPORT}")"
echo "=== TDC fetch on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "venv: ${TDC_ENV_DIR}"
"${NETRUN}" "${ENV_PY}" scripts/fetch_tdc_eval_sets.py \
    --path "${TDC_DATA_DIR}" --report "${TDC_REPORT}"
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
