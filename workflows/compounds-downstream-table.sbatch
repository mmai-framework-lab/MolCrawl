#!/bin/bash
#SBATCH --job-name=mc-cmp-ds-table
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Collect the downstream probe reports into one table on a compute node. No GPU: it
# reads the JSON the probe jobs wrote and writes a TSV.
#
#   sbatch --export=ALL,DST_ARGS='--trained DIR --untrained DIR --out FILE' \
#          workflows/compounds-downstream-table.sbatch
set -uo pipefail
: "${DST_ARGS:?set DST_ARGS to the script arguments}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== compounds downstream table on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
# shellcheck disable=SC2086 -- DST_ARGS is an argument list on purpose
"${PYTHON}" scripts/compounds_downstream_table.py ${DST_ARGS}
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
