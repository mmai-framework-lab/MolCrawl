#!/bin/bash
#SBATCH --job-name=mc-lr-stage-head
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# The learning rate logged around the head of each continued stage.
#
# No GPU: the script reads log_history out of the furthest checkpoint of each run
# directory. It runs here rather than on the login node because compute on the
# login node is not allowed here.
#
#   sbatch --export=ALL,LRH_RUNS="<dir>@15000 <dir>@30000",LRH_OUT=<...>.tsv \
#          workflows/lr-at-stage-head.sbatch
#
# LRH_RUNS is a space-separated list of DIR@HEAD_STEP. Both it and LRH_OUT are
# arguments with no default: the run directories and the report directory are
# outside every checkout, and this repository is public.
set -uo pipefail

: "${LRH_RUNS:?set LRH_RUNS to a space-separated list of DIR@HEAD_STEP}"
: "${LRH_OUT:?set LRH_OUT to the TSV to write}"
LRH_WINDOW="${LRH_WINDOW:-600}"

ARGS=()
for spec in ${LRH_RUNS}; do
  dir="${spec%@*}"
  [ -d "${dir}" ] || { echo "no such run directory: ${dir}" >&2; exit 1; }
  ARGS+=(--run "${spec}")
done

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
# A worktree holds only tracked files, so the interpreter is not inside it.
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

echo "=== learning rate at the stage head on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "tree:   ${REPO}  $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "window: ${LRH_WINDOW} step"
echo "output: ${LRH_OUT}"
"${PYTHON}" scripts/lr_at_stage_head_tsv.py \
    "${ARGS[@]}" --window "${LRH_WINDOW}" --out "${LRH_OUT}"
rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
