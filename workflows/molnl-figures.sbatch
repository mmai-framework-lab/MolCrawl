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
#          CAPTION_SIZE=<file>[,BERT_ROOT=<dir>,GPT2_ROOT=<dir>] \
#          workflows/molnl-figures.sbatch
#
# Each step runs only when the inputs it needs are given, so one figure can be redrawn
# without the others. Optional: PYTHON.
#
# Two steps were taken out on 2026-11-10: the retrieval-format figures and the overlap
# split, both drawn from the Mol-Instructions retrieval measurements that the judgment's
# 3.3 deleted. The scripts behind them are still in scripts/, with their inputs gone; this
# file no longer names the measurements, so nothing here regenerates them by habit.
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

# The size-against-compute figures, which carry the model-free floor as their dashed
# line. BERT_ROOT and GPT2_ROOT are the run roots, not a checkout, so this draws the same
# figures from a worktree as from the main checkout.
if [ -n "${BERT_ROOT:-}" ] && [ -n "${GPT2_ROOT:-}" ]; then
  [ -d "${BERT_ROOT}" ] || { echo "BERT_ROOT=${BERT_ROOT} is not a directory" >&2; exit 1; }
  [ -d "${GPT2_ROOT}" ] || { echo "GPT2_ROOT=${GPT2_ROOT} is not a directory" >&2; exit 1; }
  "${PYTHON}" scripts/molnl_scaling.py \
    --bert-root "${BERT_ROOT}" --gpt2-root "${GPT2_ROOT}" --out-dir "${FIG_DIR}" || rc=$?
fi

if [ -n "${TABLE:-}" ] && [ -n "${SIZES:-}" ] && [ -n "${CAPTION_SIZE:-}" ]; then
  "${PYTHON}" scripts/molnl_bert_size_figure.py \
    --table "${TABLE}" --sizes "${SIZES}" --out-dir "${FIG_DIR}" \
    --caption "${CAPTION_SIZE}" || rc=$?
fi

echo "=== rc=${rc} @ $(date -Is) ==="
exit "${rc}"
