#!/bin/bash
# 名前（細胞の識別子）による照合。検出力が無いと判明した方法。再現のために残すもので、判定には使わない。scripts/rna_expression_match.py を使うこと。
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=rna-identifier-match-deprecated
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
echo "NOTE: this method has no detection power -- identifiers are named per study."
echo "      Use workflows/rna-corpus-match.sbatch instead. Kept for reproduction only."
"${PY}" scripts/rna_identifier_match_deprecated.py \
  --query-dataset "${QUERY_DATASET}" --obs-id-dir "${B200_DIR}/rna/metadata_preparation_dir"
