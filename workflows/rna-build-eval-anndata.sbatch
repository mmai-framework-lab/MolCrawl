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
: "${GENE_MEDIAN_FILE:?export GENE_MEDIAN_FILE to the per-gene median pickle pretraining used}"
# The output root is stated rather than derived: this writes the artefacts a
# result is computed from, and learning_source* is gitignored, so a default
# under the checkout would put them in whichever tree the job was launched from.
OUT_DIR="${EVAL_OUT_DIR:?export EVAL_OUT_DIR to the directory this build should write to}"
echo "=== outputs: ${OUT_DIR} ==="
"${PY}" scripts/rna_build_eval_anndata.py --dataset "${QUERY_DATASET}" \
  --test-donors "${TEST_DONORS}" --hvg-n "${HVG_N}" --vocab "${VOCAB_FILE}" \
  --gene-median-file "${GENE_MEDIAN_FILE}" \
  --out-dir "${OUT_DIR}" \
  ${TISSUE_FILTER:+--tissue-filter "${TISSUE_FILTER}"} \
  ${HVG_WITHIN_VOCAB:+--hvg-within-vocab}
