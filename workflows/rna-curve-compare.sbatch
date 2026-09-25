#!/bin/bash
# Compare runs at the same steps, from the TSV the figures were drawn from.
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --job-name=rna-curve-compare
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${CURVE_TSV:?export CURVE_TSV to the TSV the figures were drawn from}"
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
mkdir -p workflows/slurm-logs
echo "=== rna-curve-compare @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
"${PYROOT}/miniconda/bin/python" scripts/rna_curve_compare.py \
  --tsv "${CURVE_TSV}" --half "${HALF:-2000}" \
  ${AT:+--at "${AT}"} ${FROM_STEP:+--from-step "${FROM_STEP}"} ${PAIRS:+--pairs "${PAIRS}"}
