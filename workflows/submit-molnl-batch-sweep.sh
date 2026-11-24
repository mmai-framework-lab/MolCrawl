#!/bin/bash
# Submit the molnl batch-size grid's arms, from the inventory, one job per arm.
#
#   LEARNING_SOURCE_DIR=<input root> MODEL_OUTPUT_ROOT=<output root> \
#   ARMS=<rt1-...arms.tsv> SUBMIT_RECORD=<record file> ARCH=bert|gpt2 \
#   [ONLY_BATCHES="256 512"] [ONLY_SIZES="small medium"] [DEPEND=<job id>] [DRY=1] \
#       workflows/submit-molnl-batch-sweep.sh
#
# ARCH=gpt2 submits one job per arm. ARCH=bert submits a chain of segments per arm,
# because an arm of this grid outruns the 96-hour limit at xl and comes close at large.
#
# ONLY_BATCHES and ONLY_SIZES narrow what goes in, which is how a grid of 68 is started
# in parts without the inventory being edited. Unset, every arm of the architecture is
# submitted.

# DEPEND holds every arm of this submission behind one job that has to end well first,
# which is how the rest of a column is queued behind the one arm that is proving a
# change. It goes on the first job of each arm only -- a BERT segment chain already
# depends on the segment before it -- so the chain is not held up twice. Without it a
# staged launch needs a session to be awake at the moment the first arm is far enough
# along, and this one is not awake for fourteen hours.
#
# The segment counts come from the §4 probes of 2026-11-20 (jobs 374100-374292), which
# measured every size at the shape this grid now uses -- so xl has a rate of its own
# rather than one scaled from large. Per-step times at global batch 2,560: small 2.517,
# medium 5.795, large 9.878, xl 19.085 seconds, each at one GPU times 4, and the
# evaluation at 1.7 / 2.8 / 3.9 / 6.8 seconds a point. Times the budget of 12,000
# steps-at-2,560 and 250 evaluation points, divided by 96 hours and rounded up, plus
# one spare:
#
#   small    8.4 h training + 0.1 h eval =   8.5 h -> 1 + 1
#   medium  19.3 h training + 0.2 h eval =  19.5 h -> 1 + 1
#   large   32.9 h training + 0.3 h eval =  33.2 h -> 1 + 1
#   xl      63.6 h training + 0.5 h eval =  64.1 h -> 1 + 1
#
# The whole grid is 6,015 GPU-hours for the 48 BERT arms on these numbers, against
# 7,395 at the 8 blocks every molnl config carried before the measurement.
#
# A segment beyond the end of a run is refused by the launcher in seconds, because it
# continues only from TIMEOUT, so a spare costs a few seconds of allocation and covers
# a slower node. xl now fits one segment on paper at 64.1 h of 96; it keeps its spare,
# and that is what the spare is for.
#
# Those figures are per arm, not per batch: the budget is fixed in blocks, so every
# batch of a size moves the same data. What is not the same is the number of optimizer
# steps it is spread over -- 120,000 at batch 256 against 7,500 at 4,096 -- and
# everything outside the forward and backward passes rides on that count. The probes
# ran at global 2,560, where one optimizer step covers 10 micro-steps at the chosen
# shape; at batch 256 the accumulation is 1, so the all-reduce and the update happen
# ten times as often for the same blocks. That increment is not in the figures above,
# which are therefore a lower bound, and it is why the two smallest batches get one
# further spare.
#
# The chain is --dependency=afterany, not afterok: a segment cut by the time limit ends
# TIMEOUT and afterok would never start the next one. The launcher continues only from
# TIMEOUT, so a segment that crashed stops the chain rather than resuming into the same
# failure. For the same reason the failure recorder goes on the last segment only: on
# every segment it would record each ordinary rollover as a failure.
#
# Each arm goes through sbatch-watched.sh, so the id is recorded outside this session.
# DRY=1 prints what would be submitted and starts nothing.
set -uo pipefail

: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the molnl input root}"
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to the output root for this grid, outside the input tree}"
: "${ARMS:?set ARMS to the derived arm inventory TSV}"
: "${SUBMIT_RECORD:?export SUBMIT_RECORD to the file to append job ids to}"
ARCH="${ARCH:-bert}"
DRY="${DRY:-0}"
DEPEND="${DEPEND:-}"
case "${DEPEND}" in
  ""|[0-9]*) ;;
  *) echo "DEPEND must be a job id, got ${DEPEND}" >&2; exit 1 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"
[ -f "${ARMS}" ] || { echo "no inventory at ${ARMS}" >&2; exit 1; }
case "${ARCH}" in
  gpt2|bert) ;;
  *) echo "ARCH must be gpt2 or bert, got ${ARCH}" >&2; exit 1 ;;
esac

# One segment per arm plus one spare. xl was 3 while its rate was large scaled by the
# parameter ratio and its estimate was 107 hours; the §4 probes measured xl itself at
# 64.1 hours, inside one 96-hour segment, so it is 2 like the rest. The batch-256 and
# batch-512 arms still get a further one below, because what the probes did not measure
# is the ten-times-more-frequent optimizer step at those batches.
declare -A SEGMENTS=([small]=2 [medium]=2 [large]=2 [xl]=2)
# The batches whose optimizer-step count is far above the rest get one more spare.
EXTRA_SEGMENT_BATCHES=" 256 512 "

only_b=""; [ -n "${ONLY_BATCHES:-}" ] && only_b=" ${ONLY_BATCHES} "
only_s=""; [ -n "${ONLY_SIZES:-}" ] && only_s=" ${ONLY_SIZES} "

# The fields are pulled out by name rather than by position, so a new column in the
# inventory does not shift what is read. awk rather than `read -a`: with IFS=tab a run
# of tabs -- which is what the empty cells of the other framework's columns are --
# collapses and shifts every later field.
rows=$(grep -v '^#' "${ARMS}" | awk -F'\t' -v want="${ARCH}" -v ob="${only_b}" -v os="${only_s}" '
  NR == 1 { for (i = 1; i <= NF; i++) at[$i] = i; next }
  $at["arch"] == want \
    && (ob == "" || index(ob, " " $at["batch"] " ") > 0) \
    && (os == "" || index(os, " " $at["size"] " ") > 0) {
    print $at["size"], $at["batch"], $at["lr_slug"], $at["config"]
  }')
[ -n "${rows}" ] || {
  echo "no ${ARCH} arm in ${ARMS}${ONLY_BATCHES:+ at batch ${ONLY_BATCHES}}${ONLY_SIZES:+ at size ${ONLY_SIZES}}" >&2
  exit 1; }

submitted=0
while read -r size batch slug cfg; do
  [ -f "${cfg}" ] || { echo "no config at ${cfg}" >&2; exit 1; }
  if [ "${ARCH}" = "gpt2" ]; then
    name="mc-molnl-gpt2-b${batch}-${size}-lr${slug}"
    gate=(); [ -n "${DEPEND}" ] && gate=(--dependency="afterok:${DEPEND}")
    args=(--job-name="${name}" "${gate[@]}"
          --export=ALL,CONFIG="${cfg}",LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}",MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}"
          workflows/molnl-batch-sweep-gpt2.sbatch)
    if [ "${DRY}" = "1" ]; then
      echo "would submit ${name}  <- ${cfg}${DEPEND:+  after ${DEPEND}}"
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
    name="mc-molnl-bert-b${batch}-${size}-lr${slug}-s${seg}"
    dep=()
    if [ -n "${prev}" ]; then
      dep=(--dependency="afterany:${prev}")
    elif [ -n "${DEPEND}" ]; then
      dep=(--dependency="afterok:${DEPEND}")
    fi
    args=(--job-name="${name}" "${dep[@]}"
          --export=ALL,MODALITY=molecule_nat_lang,CONFIG="${cfg}",SEGMENT="${seg}",LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}",MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}"
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
