#!/bin/bash
# Re-measure finished checkpoints on a fixed validation set. No training.
#
# The grid drew fresh validation rows at every eval, so its curve carries a
# scatter of 0.010-0.012. This reads the same rows for every point, so the
# points can be put beside each other at the width of the differences between
# them. Nothing is written into the runs: the only output is the TSV.
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=rna-revalidate-gpt2
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the rna learning_source root}"
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to the runs root the grid wrote to}"
: "${REVAL_CONFIG:?export REVAL_CONFIG to a config that names the rna data}"
: "${REVAL_OUT:?export REVAL_OUT to the TSV to write, outside every input tree}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
export LEARNING_SOURCE_DIR MODEL_OUTPUT_ROOT
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HOME="${MODEL_OUTPUT_ROOT}/_cache/hf" XDG_CACHE_HOME="${MODEL_OUTPUT_ROOT}/_cache/xdg"
export TMPDIR="${MODEL_OUTPUT_ROOT}/_cache/tmp"
export LD_LIBRARY_PATH="${PYROOT}/miniconda/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "${TMPDIR}" "${HF_HOME}" "${XDG_CACHE_HOME}" workflows/slurm-logs

echo "=== rna-revalidate-gpt2 @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
echo "  runs root: ${MODEL_OUTPUT_ROOT}/gpt2-grid"
echo "  out: ${REVAL_OUT}"
"${PYROOT}/miniconda/bin/python" scripts/rna_revalidate_gpt2.py \
  --config "${REVAL_CONFIG}" \
  --runs-root "${MODEL_OUTPUT_ROOT}/gpt2-grid" \
  --out "${REVAL_OUT}" \
  --rows "${REVAL_ROWS:-0}" \
  --batch-size "${REVAL_BATCH:-16}" \
  ${REVAL_CHECKPOINTS:+--checkpoints "${REVAL_CHECKPOINTS}"} \
  ${ONLY:+--only "${ONLY}"}
echo "=== done @ $(date -Is) ==="
