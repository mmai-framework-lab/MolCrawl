#!/bin/bash
#SBATCH --job-name=mc-mlm-baselines
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# Recompute the MLM judgement baselines from the data. No model involved; --gpus=1
# only because `gpu` is the sole partition here.
#
# BASELINE_TARGETS is one target per line, "<dataset dir>|<max_length>|<label>|<basename>".
# It used to be two fixed variables for the two data forms the 2026-08 arms used,
# which could not express "the same build before and after the shuffle" -- the
# comparison the baselines actually need, since the figures in circulation were
# measured on the pre-shuffle build and the runs are scored on the post-shuffle one.
#
#   sbatch --export=ALL,OUT_DIR=<dir>,BASELINE_TARGETS="<dir>|1024|packed|packed" \
#          workflows/mlm-baselines.sbatch
#
# LEARNING_SOURCE_DIR is exported because the script imports models/bert/main.py for
# resolve_eval_split and subsample_eval_split, which is what keeps the slice scored
# here identical to the one the runs evaluate on.
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
: "${BASELINE_TARGETS:?export BASELINE_TARGETS as <dir>|<max_length>|<label>|<basename> lines}"
: "${OUT_DIR:?export OUT_DIR}"
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR}"
export LEARNING_SOURCE_DIR

if [ -x "${REPO}/miniconda/bin/python" ]; then PYROOT="${REPO}"; else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"; fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON}; export PYTHON" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "${REPO}" || exit 1
mkdir -p "${OUT_DIR}" workflows/slurm-logs

echo "=== MLM baselines ${SLURM_JOB_ID} on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "code: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
rc=0
while IFS='|' read -r DIR LEN LABEL BASE; do
    [ -n "${DIR:-}" ] || continue
    echo "--- ${LABEL} (${DIR}, max_length ${LEN}) @ $(date -Is)"
    "${PYTHON}" scripts/mlm_lookup_baselines.py \
        --dataset-dir "${DIR}" --max-length "${LEN}" \
        --label "${LABEL}" --output "${OUT_DIR}/baselines_${BASE}.json" || rc=$?
done <<< "${BASELINE_TARGETS}"
echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
