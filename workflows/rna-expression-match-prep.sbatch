#!/bin/bash
# 発現による照合の前提を確かめる。ENSG の版番号・カウント 0・並び順・型を両側の実物で見て、陽性対照を相手側の読み込み経路に通す。
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=rna-expression-match-prep
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
: "${QUERY_DATASET:?export QUERY_DATASET to the census dataset id to examine}"
"${PY}" scripts/rna_expression_match_prep.py \
  --corpus-h5ad-dir "${B200_DIR}/rna/download_dir" \
  --query-dataset "${QUERY_DATASET}" --control-cells "${CONTROL_CELLS:-300}"
