#!/bin/bash
#SBATCH --job-name=mc-gen-zs-rescore-all
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# The same rescoring across a campaign's runs, one after another in one job.
#
# One job rather than twenty-one because the variant mode takes about four
# minutes a run: the queue wait would exceed the work. The window mode is not
# run this way -- at four and a half hours a run it needs its own allocation
# per subset.
#
# Required: RUNS_ROOT, RUN_GLOB, CSV, TOKENIZER, INPUT_FORM, POSITIONS, OUT_DIR.
# Optional: MAX_VARIANTS, VARIANT_BATCH, POSITION_BATCH, BF16, CHROMS.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
for v in RUNS_ROOT RUN_GLOB CSV TOKENIZER INPUT_FORM POSITIONS OUT_DIR; do
    eval ": \"\${$v:?set $v}\""
done

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "${OUT_DIR}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== ${INPUT_FORM} / ${POSITIONS} ==="
rc=0
n=0
for d in "${RUNS_ROOT}"/${RUN_GLOB}/; do
    [ -d "${d}" ] || continue
    run=$(basename "${d}")
    subset=$(echo "${run}" | sed 's/^bert-small-//;s/-w1026$//')
    out="${OUT_DIR}/${subset}.json"
    if [ -s "${out}" ]; then echo "  ${subset}: 既にある、飛ばす"; continue; fi
    echo ""; echo "---- ${subset} ----"
    "${PY}" scripts/clinvar_zeroshot_bert_rescore.py \
        --run-dir "${d%/}" --csv "${CSV}" --tokenizer "${TOKENIZER}" \
        --input-form "${INPUT_FORM}" --positions "${POSITIONS}" \
        --chroms "${CHROMS:-21,22,X,Y}" \
        ${MAX_VARIANTS:+--max-variants "${MAX_VARIANTS}"} \
        --variant-batch "${VARIANT_BATCH:-64}" \
        --position-batch "${POSITION_BATCH:-512}" \
        ${BF16:+--bf16} \
        --out "${out}" || rc=1
    n=$((n + 1))
done
echo ""; echo "---- ${n} 本 / rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
exit ${rc}
