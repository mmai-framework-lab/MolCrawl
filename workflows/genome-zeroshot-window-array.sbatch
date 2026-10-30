#!/bin/bash
#SBATCH --job-name=mc-gen-zs-window
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-20
#SBATCH --output=workflows/slurm-logs/%x-%A_%a.out
#
# BERT zero-shot over the whole window, one subset per array task.
#
# The window mode masks every base in the window in turn, so it is 1,024 times
# the work of the variant mode: one measured subset took 16,248 s at 1.314
# variants a second. That is why this is an array rather than a loop in one job
# -- a loop would need four days of wall clock in a single allocation, and a
# single failure would take the other twenty with it.
#
# The array also gives the queue behaviour that was asked for: all twenty-one
# are submitted at once, Slurm starts the ones it can place, and the rest wait
# their turn without any of them being dropped.
#
# A subset whose output file is already there is skipped, so a run that was
# measured before this array existed is not paid for twice. FORCE=1 rescores it.
#
# Required: RUNS_ROOT, CSV, TOKENIZER, OUT_DIR.
# Optional: RUN_GLOB, CHROMS, POSITION_BATCH, FORCE, NO_BF16.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
for v in RUNS_ROOT CSV TOKENIZER OUT_DIR; do
    eval ": \"\${$v:?set $v}\""
done

if [ -x "./miniconda/bin/python" ]; then PYROOT="$PWD"
else PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"; fi
PY="${PYROOT}/miniconda/bin/python"
[ -x "${PY}" ] || { echo "no python under ${PYROOT}/miniconda -- set PYROOT" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "${OUT_DIR}"

# The run list is built the same way in every task, so the index is stable.
# C collation, because the shell's locale decides whether seed1 sorts before
# seed10 and the index has to mean the same thing in the submitted list.
mapfile -t RUNS < <(ls -d "${RUNS_ROOT}"/${RUN_GLOB:-bert-small-*-w1026}/ 2>/dev/null \
                    | sed 's:/$::' | LC_ALL=C sort)
[ "${#RUNS[@]}" -gt 0 ] || { echo "no runs matched under ${RUNS_ROOT}" >&2; exit 1; }

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${IDX}" -ge "${#RUNS[@]}" ]; then
    echo "task ${IDX} is past the end of a ${#RUNS[@]} run list -- nothing to do"
    exit 0
fi

RUN_DIR="${RUNS[${IDX}]}"
SUBSET=$(basename "${RUN_DIR}" | sed 's/^bert-small-//;s/-w1026$//')
OUT="${OUT_DIR}/window-${SUBSET}.json"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-none} task=${IDX} @ $(date -Is) ==="
echo "=== ${SUBSET}  (${#RUNS[@]} 本のうち ${IDX}) ==="
echo "=== wrapped / window -> ${OUT} ==="

# bf16 is what the measured subset used; NO_BF16=1 falls back to fp32.
BF16_FLAG="--bf16"
[ -n "${NO_BF16:-}" ] && BF16_FLAG=""

# Skip only when both halves are there. The first pass wrote the aggregate
# alone, so a subset with an aggregate and no per-variant scores is unfinished
# work, not work already done, and the interval comparison needs the rows.
PREDS="${OUT%.json}-predictions.jsonl"
if [ -s "${OUT}" ] && [ -s "${PREDS}" ] && [ -z "${FORCE:-}" ]; then
    echo "既にある、飛ばす: ${OUT}"
    exit 0
fi
if [ -s "${OUT}" ] && [ ! -s "${PREDS}" ]; then
    echo "集計はあるが変異ごとの値が無い、採点し直す: ${OUT}"
fi

"${PY}" scripts/clinvar_zeroshot_bert_rescore.py \
    --run-dir "${RUN_DIR}" --csv "${CSV}" --tokenizer "${TOKENIZER}" \
    --input-form wrapped --positions window \
    --chroms "${CHROMS:-21,22,X,Y}" \
    --position-batch "${POSITION_BATCH:-512}" \
    ${BF16_FLAG} \
    --out "${OUT}"
rc=$?
echo ""; echo "---- ${SUBSET} rc=${rc} ----"
echo "=== done @ $(date -Is) ==="
exit ${rc}
