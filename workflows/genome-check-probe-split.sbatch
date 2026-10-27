#!/bin/bash
#SBATCH --job-name=mc-gen-probe-split
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Read the linear probe's folds back out of its own output: the counts on each
# side of each fold, and the overall AUROC recomputed from the per-variant
# predictions.
#
# No model is loaded and the GPU is not used; gpu is the only partition with
# compute nodes here.
#
# Required: RESULTS_ROOT. Optional: PROBE_DIRS, RECOMPUTE.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RESULTS_ROOT:?set RESULTS_ROOT to the _results directory holding the probe outputs}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PROBE_DIRS="${PROBE_DIRS:-clinvar-probe-bert-w1026-v2 clinvar-probe-gpt2-ep9 clinvar-probe-gpt2}"
ARGS=()
for d in ${PROBE_DIRS}; do ARGS+=(--probe-dir "${RESULTS_ROOT}/${d}"); done

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
"${PY}" scripts/check_probe_split.py "${ARGS[@]}" \
    --recompute "${RECOMPUTE:-mammal_centered}"
rc=$?
echo ""; echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
