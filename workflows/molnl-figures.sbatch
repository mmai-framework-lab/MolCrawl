#!/bin/bash
#SBATCH --job-name=molnl-figures
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
# The molecule_nat_lang figures and the tables under them, in one place.
#
# No GPU is requested -- it reads TSVs, JSONs and checkpoint headers -- but compute does
# not belong on the login node here.
#
#   sbatch --export=ALL,FIG_DIR=<dir>,TABLE=<all-curves tsv>,SIZES=<model sizes json>,\
#          RETRIEVAL_DIR=<formats dir>,BASELINE_DIR=<first measurements dir>,\
#          PAIR_INDEX=<pair index json>,CAPTION_SIZE=<file>,CAPTION_RETRIEVAL=<file> \
#          workflows/molnl-figures.sbatch
#
# Each step runs only when the inputs it needs are given, so one figure can be redrawn
# without the others. Optional: SPLIT_TSV (where the overlap split is written), PYTHON.
set -eo pipefail

: "${FIG_DIR:?set FIG_DIR to the directory the figures are written into}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -x "${REPO}/miniconda/bin/python" ]; then
  PYROOT="${REPO}"
else
  PYROOT="$(cd "$(git -C "${REPO}" rev-parse --git-common-dir)/.." && pwd)"
fi
PYTHON="${PYTHON:-${PYROOT}/miniconda/bin/python}"
[ -x "${PYTHON}" ] || {
  echo "no interpreter at ${PYTHON}; export PYTHON=<main checkout>/miniconda/bin/python" >&2; exit 1; }

export PYTHONPATH="${REPO}:${REPO}/scripts:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd "${REPO}"; mkdir -p workflows/slurm-logs "${FIG_DIR}"

echo "=== molnl figures on ${SLURMD_NODENAME} @ $(date -Is) ==="
echo "=== figures: ${FIG_DIR} ==="
rc=0

if [ -n "${RETRIEVAL_DIR:-}" ] && [ -n "${BASELINE_DIR:-}" ] && [ -n "${CAPTION_RETRIEVAL:-}" ]; then
  for metric in recall@1 mrr; do
    "${PYTHON}" scripts/molnl_retrieval_format_figure.py \
      --formats-dir "${RETRIEVAL_DIR}" --baseline-dir "${BASELINE_DIR}" \
      --metric "${metric}" --out-dir "${FIG_DIR}" \
      --caption "${CAPTION_RETRIEVAL}" || rc=$?
  done
fi

if [ -n "${TABLE:-}" ] && [ -n "${SIZES:-}" ] && [ -n "${CAPTION_SIZE:-}" ]; then
  "${PYTHON}" scripts/molnl_bert_size_figure.py \
    --table "${TABLE}" --sizes "${SIZES}" --out-dir "${FIG_DIR}" \
    --caption "${CAPTION_SIZE}" || rc=$?
fi

if [ -n "${RETRIEVAL_DIR:-}" ] && [ -n "${PAIR_INDEX:-}" ]; then
  "${PYTHON}" scripts/molnl_retrieval_overlap_split.py \
    --ranks-dir "${RETRIEVAL_DIR}" --pair-index "${PAIR_INDEX}" \
    --out "${SPLIT_TSV:-${FIG_DIR}/molnl-retrieval-overlap-split.tsv}" || rc=$?
fi

echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
