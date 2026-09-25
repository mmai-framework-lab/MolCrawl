#!/bin/bash
# Draw the BERT grid loss curves, and write the numbers they were drawn from.
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --job-name=rna-bert-loss-curves
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${BERT_RUNS_ROOT:?export BERT_RUNS_ROOT to the BERT output root}"
: "${CURVE_OUT_DIR:?export CURVE_OUT_DIR to where the TSV and figures should go}"
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 MPLCONFIGDIR="${CURVE_OUT_DIR}/.mpl"
mkdir -p workflows/slurm-logs "${CURVE_OUT_DIR}" "${MPLCONFIGDIR}"
echo "=== rna-bert-loss-curves @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
"${PYROOT}/miniconda/bin/python" scripts/rna_bert_loss_curves.py \
  --runs-root "${BERT_RUNS_ROOT}" --out-dir "${CURVE_OUT_DIR}" \
  ${BERT_LOG_DIR:+--log-dir "${BERT_LOG_DIR}"} \
  ${METRIC:+--metric "${METRIC}"} ${BASELINE:+--baseline "${BASELINE}"}
