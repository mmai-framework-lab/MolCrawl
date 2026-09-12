#!/bin/bash
# 評価用 AnnData と JSONL を 1 度の census 読みから作る。前提ブロック（スクリプト先頭）を読んでから投入すること。
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --job-name=rna-build-eval-anndata
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${B200_DIR:?export B200_DIR to the learning_source root holding rna/download_dir and rna/metadata_preparation_dir}"
: "${RNA_WORK_DIR:?export RNA_WORK_DIR to a writable directory for intermediates (outside any input tree)}"
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-${RNA_WORK_DIR}/tmp}"
mkdir -p "${TMPDIR}" "${RNA_WORK_DIR}" workflows/slurm-logs
echo "=== ${SLURM_JOB_NAME:-job} @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
: "${QUERY_DATASET:?export QUERY_DATASET to the census dataset id}"
: "${TEST_DONORS:?export TEST_DONORS as a comma-separated donor list for the test side}"
: "${HVG_N:?export HVG_N (1024 for the same-budget control, 2000 for the practical one)}"
: "${VOCAB_FILE:?export VOCAB_FILE to the tokenizer vocabulary pickle}"
"${PY}" scripts/rna_build_eval_anndata.py --dataset "${QUERY_DATASET}" \
  --test-donors "${TEST_DONORS}" --hvg-n "${HVG_N}" --vocab "${VOCAB_FILE}" \
  --out-dir "${RNA_WORK_DIR}/eval" \
  ${TISSUE_FILTER:+--tissue-filter "${TISSUE_FILTER}"} \
  ${HVG_WITHIN_VOCAB:+--hvg-within-vocab}
