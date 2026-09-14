#!/bin/bash
# 1 ブロックあたりの細胞数、pad が占める位置の割合、境界が参照されるかを数える。
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --job-name=rna-block-composition
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${RNA_BIN_DIR:?export RNA_BIN_DIR to the training_ready_bin directory}"
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
mkdir -p workflows/slurm-logs
echo "=== rna-block-composition @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
"${PY}" scripts/rna_block_composition.py --bin-dir "${RNA_BIN_DIR}" \
  --boundary-id "${BOUNDARY_ID:-0}" --rows "${ROWS:-20000}"
