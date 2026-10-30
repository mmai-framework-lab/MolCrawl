#!/bin/bash
#SBATCH --job-name=mc-gen-probe-fold
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# The linear probe's per-chromosome AUROCs and their count-weighted mean, one
# row per measurement, for the two model series and the model-free control.
#
# The BERT series is read from clinvar-probe-bert-w1026-v2, not from
# clinvar-probe-bert-w1026: the two differ (same checkpoints, up to about 3e-4
# apart) and the reported numbers came from v2.
#
# The control uses no model, so it is one set of values rather than 21. Its
# model_free block is identical in every probe.json the campaign wrote, and the
# job prints how many distinct blocks it found so that stays checkable.
#
# The GPU is not used. gpu is the only partition with compute nodes here.
#
# Required: RESULTS_ROOT. Optional: OUT, PROBE_BERT, PROBE_GPT2, EXPECT_SUBSETS.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${RESULTS_ROOT:?set RESULTS_ROOT to learning_source_genome_runs/_results}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PROBE_BERT="${PROBE_BERT:-${RESULTS_ROOT}/clinvar-probe-bert-w1026-v2}"
PROBE_GPT2="${PROBE_GPT2:-${RESULTS_ROOT}/clinvar-probe-gpt2-ep9}"
OUT="${OUT:-${RESULTS_ROOT}/analyses/ep9/probe-fold-21.tsv}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="

"${PY}" scripts/clinvar_probe_fold_table.py \
    --series "bert-w1026:${PROBE_BERT}" \
    --series "gpt2-ep9:${PROBE_GPT2}" \
    --control-from "${PROBE_BERT}" \
    --expect-subsets "${EXPECT_SUBSETS:-21}" \
    --verify-predictions "bert-w1026:${PROBE_BERT}" \
    --out-tsv "${OUT}"
rc=$?
echo ""; echo "---- rc=${rc} ----"
exit ${rc}
