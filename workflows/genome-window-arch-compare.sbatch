#!/bin/bash
#SBATCH --job-name=mc-gen-arch-window
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# The architecture comparison with one scoring formula on both sides, beside the
# pairing the earlier table reported, so the change from fixing the formula can
# be read off rather than inferred.
#
# Refuses below 21 window scorings: a comparison built from a partial set reads
# as a complete one.
#
# The GPU is not used. gpu is the only partition with compute nodes here.
#
# Required: RESULTS_ROOT. Optional: WINDOW_DIR, OUT_DIR, EXPECT_SUBSETS.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RESULTS_ROOT:?set RESULTS_ROOT to learning_source_genome_runs/_results}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

WINDOW_DIR="${WINDOW_DIR:-${RESULTS_ROOT}/analyses/ep9}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/analyses/ep9}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="

"${PY}" scripts/clinvar_window_arch_compare.py \
    --window-dir "${WINDOW_DIR}" \
    --bert-variant-dir "${RESULTS_ROOT}/clinvar-zeroshot-bert-w1026" \
    --gpt2-dir "${RESULTS_ROOT}/clinvar-zeroshot-gpt2-ep9" \
    --expect-subsets "${EXPECT_SUBSETS:-21}" \
    --out-tsv "${OUT_DIR}/arch-window-vs-variant.tsv" \
    --out-paired-tsv "${OUT_DIR}/arch-window-vs-variant-paired.tsv" \
    --out-md "${OUT_DIR}/arch-window-vs-variant.md"
rc=$?
echo ""; echo "---- rc=${rc} ----"
exit ${rc}
