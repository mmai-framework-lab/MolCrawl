#!/bin/bash
# Execute every grid config and check the seven things before submitting.
# Reading the files does not answer it: they inherit through `import *` and take
# their output directory from the environment.
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --job-name=rna-check-gpt2-grid
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the rna learning_source root}"
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to the runs root the grid will write to}"
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
mkdir -p workflows/slurm-logs
echo "=== rna-check-gpt2-grid @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
"${PYROOT}/miniconda/bin/python" scripts/rna_check_gpt2_grid.py
