#!/bin/bash
# 完全一致では外れる「近い重複」を、検出遺伝子の集合の重なりで見る。
# 似た細胞どうしは重複でなくても集合がよく重なるので、重なりが高い対が
# あること自体は重複の証拠にならない。陽性対照（コーパスの細胞そのもの）と
# 基準（別の後発データセット）を並べ、陽性対照が 1.0 に届かなければ判定しない。
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --job-name=rna-near-duplicate
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
: "${MISSING_IDS:?export MISSING_IDS to the .npy of soma_joinids absent from download_dir (see rna-fetch-missing.sbatch)}"
"${PY}" scripts/rna_near_duplicate.py \
  --corpus-h5ad-dir "${B200_DIR}/rna/download_dir" \
  --obs-id-dir "${B200_DIR}/rna/metadata_preparation_dir" \
  --missing-ids "${MISSING_IDS}" \
  --query-sample "${QUERY_SAMPLE:-2000}" \
  --out "${RNA_WORK_DIR}/near_dup"
