#!/bin/bash
# Submit the compounds batch-size grid's arms, one job per arm, from the inventory.
#
#   LEARNING_SOURCE_DIR=<input root> MODEL_OUTPUT_ROOT=<output root> \
#   ARMS=<rt1-...arms.tsv> SUBMIT_RECORD=<record file> [ARCH=gpt2] \
#   [ONLY_BATCHES="256 512"] [DRY=1] \
#       workflows/submit-compounds-batch-sweep.sh
#
# ONLY_BATCHES submits just the arms at those global batches. The inventory holds
# every arm of the grid, including ones that have already run: the 2026-11-20
# directive added batch 256 to a GPT-2 column whose other 16 arms were finished and
# are explicitly not to be re-run, so submitting by architecture alone would start
# them again. Unset, every arm of the architecture is submitted.
#
# ARCH=gpt2 submits one job per arm; ARCH=bert submits a chain of segments per arm,
# because a BERT arm of this grid outruns the 96-hour limit at three of the four sizes.
#
# The segment counts come from the measured bf16 step time of each size at the shape
# the configs now carry (small 3.13 s/step, medium 7.35, large 12.14, xl 23.70 at
# global batch 2,560), times the 45,000-step-equivalent budget, plus the cost of 225
# evaluation points at the grid's evaluation micro-batch of 32, divided by 96 hours and
# rounded up, plus one spare:
#
#   small   39.1 h training + 0.4 h eval =  39.5 h -> 1 + 1
#   medium  91.9 h training + 5.3 h eval =  97.2 h -> 2 + 1
#   large  151.8 h training + 2.9 h eval = 154.7 h -> 2 + 1
#   xl     296.3 h training + 2.4 h eval = 298.7 h -> 4 + 1
#
# The eval figures are the 150-point costs measured on jobs 150891-150894 scaled to 225
# points and to the evaluation micro-batch of 32 -- medium and large scored at 160 and
# 64 then, so their eval points now cost 5x and 2x what was measured. A segment beyond
# the end of a run is refused by the launcher in seconds, so a spare costs a few
# seconds of allocation and covers a slower node.
#
# Those figures are per arm, not per batch: the budget is fixed in columns, so every
# batch of a size moves the same data and the training time above is the same at 256 as
# at 4096. What is not the same is the number of optimizer steps the time is spread
# over -- 450,000 at batch 256 against 28,125 at 4096 -- and the per-step overhead
# outside the forward and backward passes rides on that count. It has not been
# measured at this shape, so the arms at batch 256 get one further spare segment.
#
# The chain is --dependency=afterany, not afterok: a segment cut by the time limit ends
# TIMEOUT and afterok would never start the next one. The launcher continues only from
# TIMEOUT, so a segment that crashed stops the chain rather than resuming into the same
# failure. For the same reason the failure recorder goes on the last segment only: on
# every segment it would record each ordinary rollover as a failure.
#
# Each arm goes through sbatch-watched.sh, so the id is recorded. DRY=1 prints what
# would be submitted.
set -uo pipefail

: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the packed compounds input root}"
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to a runs root outside the input tree}"
: "${ARMS:?set ARMS to the derived arm inventory TSV}"
: "${SUBMIT_RECORD:?export SUBMIT_RECORD to the file to append job ids to}"
ARCH="${ARCH:-gpt2}"
DRY="${DRY:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"
[ -f "${ARMS}" ] || { echo "no inventory at ${ARMS}" >&2; exit 1; }

case "${ARCH}" in
  gpt2|bert) ;;
  *) echo "ARCH must be gpt2 or bert, got ${ARCH}" >&2; exit 1 ;;
esac

# One GPU's column count was the open question that held the BERT arms back. The
# directive of 2026-11-19 on that one question replaced 19c §8 and settled it, and the
# configs were regenerated from it, so bert is submitted like gpt2 now.
declare -A SEGMENTS=([small]=2 [medium]=3 [large]=3 [xl]=5)
# The batches whose step count is far above the rest get one more spare (see above).
EXTRA_SEGMENT_BATCHES=" 256 "

submitted=0
# The fields are pulled out by name rather than by position, so a new column in the
# inventory does not shift what is read. awk rather than `read -a`: IFS=tab treats a
# tab as whitespace, so a run of tabs -- which is what the empty cells of the other
# framework's columns are -- would collapse and shift every later field.
only=""
[ -n "${ONLY_BATCHES:-}" ] && only=" ${ONLY_BATCHES} "
rows=$(grep -v '^#' "${ARMS}" | awk -F'\t' -v want="${ARCH}" -v only="${only}" '
  NR == 1 { for (i = 1; i <= NF; i++) at[$i] = i; next }
  $at["arch"] == want && (only == "" || index(only, " " $at["batch"] " ") > 0) {
    print $at["size"], $at["batch"], $at["lr_slug"], $at["config"]
  }')
[ -n "${rows}" ] || { echo "no ${ARCH} arm in ${ARMS}${ONLY_BATCHES:+ at batch ${ONLY_BATCHES}}" >&2; exit 1; }

while read -r size batch slug cfg; do
  [ -f "${cfg}" ] || { echo "no config at ${cfg}" >&2; exit 1; }
  if [ "${ARCH}" = "gpt2" ]; then
    name="mc-cmp-gpt2-${size}-b${batch}-lr${slug}"
    args=(--job-name="${name}"
          --export=ALL,CONFIG="${cfg}",LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}",MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}"
          workflows/compounds-gpt2-grid.sbatch)
    if [ "${DRY}" = "1" ]; then
      echo "would submit ${name}  <- ${cfg}"
    else
      jid=$(SUBMIT_RECORD="${SUBMIT_RECORD}" workflows/sbatch-watched.sh "${name}" -- "${args[@]}")
      echo "${jid}  ${name}"
    fi
    submitted=$((submitted + 1))
    continue
  fi

  segments="${SEGMENTS[$size]}"
  case "${EXTRA_SEGMENT_BATCHES}" in *" ${batch} "*) segments=$((segments + 1)) ;; esac
  prev=""
  for seg in $(seq 1 "${segments}"); do
    name="mc-cmp-bert-${size}-b${batch}-lr${slug}-s${seg}"
    dep=(); [ -n "${prev}" ] && dep=(--dependency="afterany:${prev}")
    args=(--job-name="${name}" "${dep[@]}"
          --export=ALL,MODALITY=compounds,CONFIG="${cfg}",SEGMENT="${seg}",LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}",MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}"
          workflows/bert-grid.sbatch)
    # The recorder belongs on the last segment: afternotok fires on TIMEOUT, which is
    # how every segment but the last is meant to end.
    if [ "${seg}" = "${segments}" ]; then watch=1; else watch=0; fi
    if [ "${DRY}" = "1" ]; then
      echo "would submit ${name}  <- ${cfg}  segment ${seg}/${segments}${prev:+ after ${prev}}$([ "${watch}" = 1 ] && echo '  [recorder]')"
      prev="<${name}>"
    else
      jid=$(SUBMIT_RECORD="${SUBMIT_RECORD}" WATCH="${watch}" \
            workflows/sbatch-watched.sh "${name}" -- "${args[@]}")
      echo "${jid}  ${name}  segment ${seg}/${segments}${prev:+ after ${prev}}"
      prev="${jid}"
    fi
    submitted=$((submitted + 1))
  done
done <<< "${rows}"

echo "${submitted} ${ARCH} job(s) $([ "${DRY}" = "1" ] && echo 'would be submitted' || echo submitted)"
