#!/bin/bash
#SBATCH --job-name=mc-gen-results-v2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# The ClinVar results table, with both GPT-2 epoch series in it.
#
# Each input is given as series:epochs:seed:path, so a row says which schedule
# and which seed it came from. Until the re-run at seed 42 is complete the
# table holds the present values, and the judgment of 2026-10-22 asks that both
# be written beside them.
#
# The nine-epoch fold analyses come from analyses/ep9 and the three-epoch ones
# from analyses/corr2, whose second correlation is measured on matched ranges.
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
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/analyses/results-v2}"
CORR2="${RESULTS_ROOT}/analyses/corr2"
EP9="${RESULTS_ROOT}/analyses/ep9"
mkdir -p "${OUT_DIR}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== out: ${OUT_DIR} ==="

"${PY}" scripts/clinvar_results_table.py \
    --zeroshot-bert "bert-w1026:9:35:${CORR2}/zeroshot-bert.json" \
    --probe-bert    "bert-w1026:9:35:${CORR2}/probe-bert.json" \
    --zeroshot-gpt2 "gpt2:3:45:${CORR2}/zeroshot-gpt2.json" \
    --probe-gpt2    "gpt2:3:45:${CORR2}/probe-gpt2.json" \
    --zeroshot-gpt2 "gpt2-ep9:9:45:${EP9}/zeroshot-gpt2-ep9.json" \
    --probe-gpt2    "gpt2-ep9:9:45:${EP9}/probe-gpt2-ep9.json" \
    --probe-json-bert "${RESULTS_ROOT}/clinvar-probe-bert-w1026-v2/probe.json" \
    --probe-json-gpt2 "${RESULTS_ROOT}/clinvar-probe-gpt2-ep9/probe.json" \
    --untrained-bert  "${RESULTS_ROOT}/clinvar-probe-bert-untrained-v2/probe.json" \
    --untrained-gpt2  "${RESULTS_ROOT}/clinvar-probe-gpt2-untrained/probe.json" \
    --out-tsv "${OUT_DIR}/clinvar-results.tsv" \
    --out-md  "${OUT_DIR}/clinvar-results.md"
rc=$?
echo ""; echo "---- rc=${rc} ----"
exit ${rc}
