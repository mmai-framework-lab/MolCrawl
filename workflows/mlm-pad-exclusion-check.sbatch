#!/bin/bash
# pad が MLM のマスク選択と損失から除外されているかを、学習と同じ collator で測る。
# 除外されていなければマスクの予算が詰め物の予測に使われる。モダリティ非依存。
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --job-name=mlm-pad-exclusion-check
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${TOKENIZER_DIR:?export TOKENIZER_DIR to the tokenizer directory to check}"
if [ -x "./miniconda/bin/python" ]; then
    PYROOT="$PWD"
else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"
fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p workflows/slurm-logs
echo "=== mlm-pad-exclusion-check @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
"${PY}" scripts/mlm_pad_exclusion_check.py \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  ${MODALITY:+--modality "${MODALITY}"} \
  --pad-id "${PAD_ID:-0}"
