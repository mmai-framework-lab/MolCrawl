#!/bin/bash
# download_dir が学習に使った細胞をどれだけ持っているかを数える。
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --job-name=rna-corpus-coverage
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
# ツリーの場所は環境変数で受ける。既定を置かず、未設定なら落とす。
: "${B200_DIR:?export B200_DIR to the learning_source root holding rna/download_dir and rna/metadata_preparation_dir}"
: "${RNA_WORK_DIR:?export RNA_WORK_DIR to a writable directory for intermediates (outside any input tree)}"
[ -d "${B200_DIR}/rna/download_dir" ] || {
    echo "B200_DIR=${B200_DIR} has no rna/download_dir under it" >&2; exit 1; }

# miniconda は worktree に無い。git の共通ディレクトリから本体の checkout を引く。
if [ -x "./miniconda/bin/python" ]; then
    PYROOT="$PWD"
else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"
fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }

export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-${RNA_WORK_DIR}/tmp}"
mkdir -p "${TMPDIR}" "${RNA_WORK_DIR}" workflows/slurm-logs
echo "=== ${SLURM_JOB_NAME:-job} @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
echo "B200_DIR=${B200_DIR}"
echo "RNA_WORK_DIR=${RNA_WORK_DIR}"
echo "python=${PY}"
"${PY}" scripts/rna_corpus_coverage.py \
  --corpus-h5ad-dir "${B200_DIR}/rna/download_dir" \
  --obs-id-dir "${B200_DIR}/rna/metadata_preparation_dir"
