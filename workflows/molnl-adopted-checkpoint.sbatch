#!/bin/bash
#SBATCH --job-name=mc-molnl-adopted-ckpt
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# 2026-11-10 judgment 5.11: is the checkpoint the retrieval scoring was given the one the
# trainer resolved as best, or the last step of the schedule?
#
#   sbatch --export=ALL,RUNS="<dir>=<ckpt> <dir>=<ckpt>",OUT=<json> \
#          workflows/molnl-adopted-checkpoint.sbatch
#
# It reads each run's own trainer_state.json and run_manifest.json. No GPU is needed; the
# partition allocates one anyway. It runs on a compute node because nothing in this
# project runs on a login node.
set -eo pipefail

: "${RUNS:?set RUNS to space separated <run dir>=<scored checkpoint> pairs}"
: "${OUT:?set OUT to the json to write}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${REPO}"; mkdir -p workflows/slurm-logs "$(dirname "${OUT}")"

RUN_ARGS=()
for r in ${RUNS}; do RUN_ARGS+=(--run "${r}"); done

echo "=== molnl adopted checkpoint on ${SLURMD_NODENAME} @ $(date -Is) ==="
rc=0
"${PYTHON}" scripts/molnl_bert_adopted_checkpoint.py "${RUN_ARGS[@]}" --out "${OUT}" || rc=$?
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
