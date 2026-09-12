#!/bin/bash
# 評価データの分割候補を出す。ドナー単位で切り、細胞数比・測定様式比・組織区分比の 3 つが揃う組を総当りで探す。
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=rna-eval-split-candidates
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
: "${QUERY_DATASET:?export QUERY_DATASET to the census dataset id to split}"
: "${VOCAB_FILE:?export VOCAB_FILE to the tokenizer vocabulary pickle}"
"${PY}" scripts/rna_eval_split_candidates.py --dataset "${QUERY_DATASET}" \
  --vocab "${VOCAB_FILE}" --out-obs "${RNA_WORK_DIR}/eval_obs.pkl" \
  ${AIRWAY_TISSUES:+--airway-tissues "${AIRWAY_TISSUES}"} \
  --test-donors "${TEST_DONORS:-3}"
