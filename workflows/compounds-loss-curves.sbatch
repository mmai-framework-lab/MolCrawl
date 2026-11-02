#!/bin/bash
#SBATCH --job-name=mc-cmp-loss-curves
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Build the long-format train/validation curve TSV on a compute node. No GPU: the
# script reads the HF trainer_state.json of each BERT run and nanoGPT's per-run csv.
# It runs here rather than on the login node because the whole history of a 45,000
# step run is read for every arm.
#
#   sbatch --export=ALL,CURVE_ARGS='--run ARCH:SIZE:LR:SEED:DIR --run ... --out FILE' \
#          workflows/compounds-loss-curves.sbatch
#
# The validation metric defaults to eval_loss_mask, which is what the compounds BERT
# runs are scored on; pass --val-metric to the script to change it.
set -uo pipefail
: "${CURVE_ARGS:?set CURVE_ARGS to the script arguments}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== compounds loss curves on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
# Globs in the argument list belong to the script, not to this shell.
set -f
# shellcheck disable=SC2086 -- CURVE_ARGS is an argument list on purpose
"${PYTHON}" scripts/build_loss_curve_tsv.py ${CURVE_ARGS}
rc=$?
set +f
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
