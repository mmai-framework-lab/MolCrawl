#!/bin/bash
#SBATCH --job-name=mc-gen-pair-values
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# The AUROCs the architecture comparison was computed from, one row per subset.
#
# The comparison was reported as a median and a range, which says which side
# won and by how much in the middle but not what either side scored, and cannot
# be checked against anything. The values are read from where the campaign
# wrote them, not recomputed.
#
# The GPU is not used. gpu is the only partition with compute nodes here.
#
# Required: RESULTS_ROOT. Optional: OUT_DIR.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RESULTS_ROOT:?set RESULTS_ROOT to learning_source_genome_runs/_results}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/analyses/ep9}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="

"${PY}" scripts/clinvar_arch_pair_values.py \
    --results-root "${RESULTS_ROOT}" \
    --out-tsv "${OUT_DIR}/arch-pair-values.tsv" \
    --out-md  "${OUT_DIR}/arch-pair-values.md"
rc=$?
echo ""; echo "---- rc=${rc} ----"
exit ${rc}
