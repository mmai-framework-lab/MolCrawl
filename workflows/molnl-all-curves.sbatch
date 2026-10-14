#!/bin/bash
#SBATCH --job-name=mc-molnl-all-curves
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Build the one-row-per-evaluation-point table for every molecule_nat_lang run.
#
# No GPU: it reads trainer_state.json and nanoGPT's logging CSVs. It still goes through
# the scheduler -- nothing computes on the login node.
#
#   sbatch --export=ALL,ARTIFACT_ROOT=<.../molecule_nat_lang>,OUT_TSV=<file.tsv>[,LOG_DIRS="a b"]\
#          workflows/molnl-all-curves.sbatch
#
# LOG_DIRS defaults to the slurm-logs of every checkout beside this one plus the two
# learning_source log directories, because the job ids come from the launchers' banners
# and the runs were launched from several worktrees over the grid's life.
set -eo pipefail

: "${ARTIFACT_ROOT:?set ARTIFACT_ROOT to the molecule_nat_lang directory}"
: "${OUT_TSV:?set OUT_TSV to the table to write}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs

if [ -z "${LOG_DIRS:-}" ]; then
  LOG_DIRS=""
  for d in "$(dirname "${PYROOT}")"/*/workflows/slurm-logs; do
    [ -d "${d}" ] && LOG_DIRS="${LOG_DIRS} ${d}"
  done
  for d in "$(dirname "${ARTIFACT_ROOT}")"*/molecule_nat_lang/logs; do
    [ -d "${d}" ] && LOG_DIRS="${LOG_DIRS} ${d}"
  done
fi
echo "=== all-curves on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "    log dirs:${LOG_DIRS}"

rc=0
# shellcheck disable=SC2086
"${PYTHON}" scripts/molnl_all_curves_tsv.py \
  --artifact-root "${ARTIFACT_ROOT}" --log-dirs ${LOG_DIRS} \
  ${RESCORED:+--rescored "${RESCORED}"} ${RESCORE_JOB:+--rescore-job "${RESCORE_JOB}"} \
  --out "${OUT_TSV}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
