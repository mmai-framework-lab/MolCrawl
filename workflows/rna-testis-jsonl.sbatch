#!/bin/bash
# testis 版の評価 JSONL を、学習時と同じ順序で作り直す。
# 生カウント順で作られた旧版は使えない（上長 2026-09-08 §2）。
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=rna-testis-jsonl
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${TESTIS_H5AD:?export TESTIS_H5AD to the Tabula Sapiens testis h5ad}"
: "${TOKENIZER_DIR:?export TOKENIZER_DIR to the RNA BERT tokenizer directory}"
: "${OUT_JSONL:?export OUT_JSONL to the path the rebuilt JSONL should be written to}"
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p workflows/slurm-logs "$(dirname "${OUT_JSONL}")"
echo "=== rna-testis-jsonl @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
echo "=== outputs: ${OUT_JSONL} ==="
"${PY}" -m molcrawl.tasks.evaluation.tabula_sapiens.prepare_jsonl \
  --h5ad-url "file://${TESTIS_H5AD}" --h5ad-path "${TESTIS_H5AD}" \
  --output-jsonl "${OUT_JSONL}" --tokenizer-dir "${TOKENIZER_DIR}"
