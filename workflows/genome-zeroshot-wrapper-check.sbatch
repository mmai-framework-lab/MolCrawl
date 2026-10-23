#!/bin/bash
#SBATCH --job-name=mc-gen-wrapcheck
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=workflows/slurm-logs/%x-%j.out
#
# Does the BERT zero-shot score move when the window is wrapped as in training?
#
# The adapter sends 1,024 bare base tokens because this tokeniser has no
# post-processor template; the model was trained on [CLS] + 1,024 + [SEP] and
# the linear probe builds that form. One run is scored both ways here, and the
# bare mode doubles as a control: if it does not reproduce the AUROC the
# campaign recorded, the wrapped number is not evidence of anything.
#
# Nothing already written is touched. This measures; it does not re-score the
# campaign.
#
# Required: CSV, CHECKPOINT, TOKENIZER. Optional: RECORDED, OUT, BATCH_SIZE, CHROMS.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${CSV:?set CSV to the scored clinvar table}"
: "${CHECKPOINT:?set CHECKPOINT to the adopted BERT checkpoint}"
: "${TOKENIZER:?set TOKENIZER to the tokenizer directory}"

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} @ $(date -Is) ==="
echo "=== checkpoint: ${CHECKPOINT} ==="

"${PY}" scripts/clinvar_zeroshot_wrapper_check.py \
    --csv "${CSV}" --checkpoint "${CHECKPOINT}" --tokenizer "${TOKENIZER}" \
    ${RECORDED:+--recorded-metrics "${RECORDED}"} \
    --chroms "${CHROMS:-21,22,X,Y}" \
    --batch-size "${BATCH_SIZE:-32}" \
    ${OUT:+--out "${OUT}"}
rc=$?
echo ""; echo "---- rc=${rc} ----"
exit ${rc}
