#!/bin/bash
# 評価側の切り出しが、学習時と同じ順序（中央値スケール）になっているかを測る。生カウント順との顔ぶれの一致率と、空間ごとの発現量の割合を出す。
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=rna-rank-basis-check
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
: "${GENE_MEDIAN_FILE:?export GENE_MEDIAN_FILE to the per-gene median pickle pretraining used}"
: "${QUERY_DATASET:?export QUERY_DATASET to the census dataset id to examine}"
"${PY}" scripts/rna_rank_basis_check.py --dataset "${QUERY_DATASET}" \
  --gene-median-file "${GENE_MEDIAN_FILE}" --sample "${SAMPLE:-5000}"
