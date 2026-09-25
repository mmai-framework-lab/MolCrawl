#!/bin/bash
# The model-free control for the cell-type probe: HVG principal components.
#
# Two conditions. same-budget takes the HVGs from inside the model's vocabulary
# and holds the count to the model's window, so the comparison is between
# representations. practical takes them from all genes, so the comparison is
# about what to use for the task. Both read the same cells and the same split.
#
# No GPU: this is a logistic regression on principal components.
#SBATCH --partition=gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --job-name=rna-hvg-control
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#SBATCH --error=workflows/slurm-logs/%x-%j.out
set -e
: "${CONTROL_ROOT:?export CONTROL_ROOT to the directory holding the condition subdirectories}"
: "${CONTROL_OUT_DIR:?export CONTROL_OUT_DIR to where the results go, outside every input tree}"
if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"; else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p workflows/slurm-logs "${CONTROL_OUT_DIR}"

echo "=== rna-hvg-control @ $(date -Is) on ${SLURMD_NODENAME:-$(hostname)} ==="
echo "=== outputs: ${CONTROL_OUT_DIR} ==="
for COND in ${CONTROL_CONDITIONS:-same-budget practical}; do
    H5="${CONTROL_ROOT}/${COND}/eval.h5ad"
    [ -f "${H5}" ] || { echo "no h5ad for ${COND}: ${H5}" >&2; exit 1; }
    echo "--- ${COND} @ $(date -Is)"
    "${PY}" scripts/rna_hvg_control.py \
        --h5ad "${H5}" --out-dir "${CONTROL_OUT_DIR}/${COND}" \
        ${MIN_CELLS:+--min-cells "${MIN_CELLS}"}
done
echo "=== done @ $(date -Is) ==="
