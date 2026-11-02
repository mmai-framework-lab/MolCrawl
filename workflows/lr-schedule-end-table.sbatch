#!/bin/bash
#SBATCH --job-name=mc-lr-end-table
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# How each modality's learning rate ends. No GPU: the script reads config text and
# trainer_state.json. It runs here rather than on the login node because an rna
# trainer_state carries 120,960 steps of log_history and sits on Lustre.
#
#   sbatch --export=ALL,LREND_ARGS='--repo DIR --bert M DIR CFG ... --out FILE' \
#          workflows/lr-schedule-end-table.sbatch
set -uo pipefail
: "${LREND_ARGS:?set LREND_ARGS to the script arguments}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs
echo "=== lr schedule end table on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
# shellcheck disable=SC2086 -- LREND_ARGS is an argument list on purpose
"${PYTHON}" scripts/lr_schedule_end_table.py ${LREND_ARGS}
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
