#!/bin/bash
#SBATCH --job-name=mc-gen-wloss
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Account for the bases each chunk length drops. Rebuilding at 1,024 uses 1.37%
# fewer bases than the 512 build, and the remainder of a full Phase 2 line
# explains only 0.195% of that. This walks the raw sequences to find the rest.
#
# Required: GENOME_SOURCE_ROOT. Optional: SUBSETS (default: all).

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [ -n "${SUBSETS:-}" ]; then
    LIST="${SUBSETS//,/ }"
else
    LIST=$(for d in "${GENOME_SOURCE_ROOT}/genome_sequence"/*/; do
               [ -d "${d}/raw_files" ] && basename "${d}"; done | sort)
fi

echo "=== node=$(hostname) subsets=$(echo ${LIST} | wc -w) ==="
for s in ${LIST}; do
    echo ""
    echo "########## ${s} ##########"
    "${PY}" scripts/analyze_genome_window_loss.py \
        "${GENOME_SOURCE_ROOT}/genome_sequence/${s}/raw_files" --chunks 510,1022
done
echo ""
echo "=== done @ $(date -Is) ==="
