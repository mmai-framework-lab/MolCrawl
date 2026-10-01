#!/bin/bash
#SBATCH --job-name=mc-gen-groupcorr
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Does the shape of a subset explain its ClinVar AUROC?
#
# Three candidates, none needing a model: how many assemblies the subset drew,
# how long its contigs are, and how much of it windowing threw away.
#
# WHY THIS IS TAKEN INSIDE EACH GROUP. Assembly count is 41-112 for
# eukaryote_matched and 473-885 for global_random; the ranges do not overlap, so
# pooled it is almost a label for the group. Any group difference would reappear
# as a correlation with it, and the answer would be that the two groups differ
# -- the question, not the answer. If the relationship does not hold inside a
# group, assembly count explains nothing.
#
# mammal_centered is one subset and is left out: one point has no correlation.
#
# No GPU work happens here. It runs on the gpu partition because that is where
# compute nodes are, and nothing in this repo runs on the login node.
#
# Required: RESULTS_ROOT. Optional: OUT_DIR, SUBSETS_DIR, WINDOW_LOSS.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RESULTS_ROOT:?set RESULTS_ROOT to learning_source_genome_runs/_results}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/analyses/results}"
mkdir -p "${OUT_DIR}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== out: ${OUT_DIR} ==="

"${PY}" scripts/clinvar_group_correlations.py \
    --results-tsv "${OUT_DIR}/clinvar-results.tsv" \
    --subsets-dir "${SUBSETS_DIR:-$PWD/assets/genome_species_list/subsets}" \
    --window-loss "${WINDOW_LOSS:-${RESULTS_ROOT}/analyses/window-loss.json}" \
    --out-tsv "${OUT_DIR}/group-correlations.tsv" \
    --out-md  "${OUT_DIR}/group-correlations.md"
rc=$?
echo ""; echo "---- rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
exit ${rc}
