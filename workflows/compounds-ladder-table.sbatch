#!/bin/bash
#SBATCH --job-name=mc-cmp-ladder-table
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Build the compounds ladder table on a compute node. No GPU is requested: the
# script reads trainer_state.json, training_args.json and safetensors headers. It
# runs here rather than on the login node because nothing computes there, and
# because the safetensors headers sit on Lustre next to multi-gigabyte shards.
#
#   sbatch --export=ALL,LADDER_ARGS='--bert-root DIR --gpt2-root DIR --out FILE' \
#          workflows/compounds-ladder-table.sbatch
set -uo pipefail
: "${LADDER_ARGS:?set LADDER_ARGS to the script arguments}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== compounds ladder table on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
# shellcheck disable=SC2086 -- LADDER_ARGS is an argument list on purpose
"${PYTHON}" scripts/compounds_ladder_table.py ${LADDER_ARGS}
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
