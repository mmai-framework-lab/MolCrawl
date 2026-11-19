#!/bin/bash
# Submit the re-measurement of the adopted compounds BERT arms, one job per point.
#
#   LEARNING_SOURCE_DIR=<input root> MODEL_OUTPUT_ROOT=<output root> \
#   POINTS=<rt1-...reval-points.tsv> SUBMIT_RECORD=<record file> [DRY=1] \
#       workflows/submit-compounds-reval.sh
#
# 2026-11-19c §11: the existing 2,560 numbers were measured with the masked positions
# redrawn at every evaluation, and the new grid fixes them, so the two series do not
# join until the existing checkpoints are measured again under the grid's evaluation.
#
# The first point is submitted on its own and the rest wait on it finishing well.
# The eval-only path through main.py has unit tests behind it but has never run on a
# GPU, and twenty jobs failing the same way at once is the outcome worth avoiding;
# the whole set still goes into the queue in one submission.
#
# All twenty share the evaluation: 4 processes, evaluation micro-batch 32, the 10,000
# -row subset, fixed mask with seed 42. The training micro-batch is not used -- nothing
# trains -- so none of this waits on §8.
set -uo pipefail

: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the packed compounds input root}"
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to a runs root outside the input tree}"
: "${POINTS:?set POINTS to the re-measurement point inventory TSV}"
: "${SUBMIT_RECORD:?export SUBMIT_RECORD to the file to append job ids to}"
DRY="${DRY:-0}"
EVAL_BATCH="${EVAL_BATCH:-32}"
MASK_SEED="${MASK_SEED:-42}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"
[ -f "${POINTS}" ] || { echo "no point inventory at ${POINTS}" >&2; exit 1; }

RUNS="${MODEL_OUTPUT_ROOT}/compounds/bert-output"
# The measurement's own tree, outside every run it measures.
OUT_ROOT="${MODEL_OUTPUT_ROOT}/compounds/bert-output-reval-s42"
RECORDS="${OUT_ROOT}/records"

# awk rather than `read -a`: IFS=tab treats a tab as whitespace, so the empty cells of
# the not-yet-measured columns would collapse and shift every later field.
rows=$(grep -v '^#' "${POINTS}" | awk -F'\t' '
  NR == 1 { for (i = 1; i <= NF; i++) at[$i] = i; next }
  { print $at["size"], $at["step"], $at["checkpoint"] }')
[ -n "${rows}" ] || { echo "no point in ${POINTS}" >&2; exit 1; }

first=""
submitted=0
while read -r size step ckpt; do
  # The config supplies the architecture, the dataset and the evaluation; the weights
  # come from the named checkpoint, so one config per size covers all five points.
  cfg=$(ls molcrawl/tasks/pretrain/configs/compounds/bert_${size}_lr*_seed42_ext45k.py 2>/dev/null | head -1)
  [ -n "${cfg}" ] || { echo "no adopted config for ${size}" >&2; exit 1; }
  name="mc-cmp-reval-${size}-${step}"
  args=(--job-name="${name}"
        --export=ALL,LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}",MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}",CONFIG="${cfg}",CKPT="${RUNS}/${ckpt}",OUT="${OUT_ROOT}/${size}-${step}",OUT_JSON="${RECORDS}/${size}-${step}.json",EVAL_BATCH="${EVAL_BATCH}",MASK_SEED="${MASK_SEED}"
        workflows/bert-eval-only.sbatch)
  [ -n "${first}" ] && args=(--dependency=afterok:"${first}" "${args[@]}")
  if [ "${DRY}" = "1" ]; then
    echo "would submit ${name}  <- ${cfg}  ckpt ${ckpt}$([ -n "${first}" ] && echo "  after ${first}")"
    [ -z "${first}" ] && first="FIRST"
  else
    jid=$(SUBMIT_RECORD="${SUBMIT_RECORD}" workflows/sbatch-watched.sh "${name}" -- "${args[@]}")
    echo "${jid}  ${name}$([ -n "${first}" ] && echo "  after ${first}")"
    [ -z "${first}" ] && first="${jid}"
  fi
  submitted=$((submitted + 1))
done <<< "${rows}"

echo "${submitted} point(s) $([ "${DRY}" = "1" ] && echo 'would be submitted' || echo submitted)"
echo "records land in ${RECORDS}; collect them with scripts/compounds_reval_points.py --collect"
