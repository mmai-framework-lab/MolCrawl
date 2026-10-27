#!/bin/bash
#SBATCH --job-name=mc-gen-zs-rescore
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# BERT zero-shot, with the input form and the scored positions set explicitly.
#
# INPUT_FORM  wrapped = [CLS] + 1,024 + [SEP], as training and the probe use
#             bare    = the 1,024 base tokens the adapter sends today
# POSITIONS   variant = the variant position alone (2026-09-07 verdict)
#             window  = every base in turn, averaged (GPT-2's quantity)
#
# The window mode is 1,024 times the work of the variant mode. MAX_VARIANTS
# scores a prefix so its cost is measured before it is paid.
#
# Required: RUN_DIR, CSV, TOKENIZER, INPUT_FORM, POSITIONS, OUT.
# Optional: MAX_VARIANTS, VARIANT_BATCH, POSITION_BATCH, BF16, CHROMS.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
for v in RUN_DIR CSV TOKENIZER INPUT_FORM POSITIONS OUT; do
    eval ": \"\${$v:?set $v}\""
done

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== ${INPUT_FORM} / ${POSITIONS} / run ${RUN_DIR} ==="

"${PY}" scripts/clinvar_zeroshot_bert_rescore.py \
    --run-dir "${RUN_DIR}" --csv "${CSV}" --tokenizer "${TOKENIZER}" \
    --input-form "${INPUT_FORM}" --positions "${POSITIONS}" \
    --chroms "${CHROMS:-21,22,X,Y}" \
    ${MAX_VARIANTS:+--max-variants "${MAX_VARIANTS}"} \
    --variant-batch "${VARIANT_BATCH:-64}" \
    --position-batch "${POSITION_BATCH:-512}" \
    ${BF16:+--bf16} \
    --out "${OUT}"
rc=$?
echo ""; echo "---- rc=${rc} ----"
exit ${rc}
