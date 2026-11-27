#!/bin/bash
# Submit the molnl batch-size grid's arms, from the inventory, one job per arm.
#
#   LEARNING_SOURCE_DIR=<input root> MODEL_OUTPUT_ROOT=<output root> \
#   ARMS=<rt1-...arms.tsv> SUBMIT_RECORD=<record file> ARCH=bert|gpt2 \
#   [ONLY_BATCHES="256 512"] [ONLY_SIZES="small medium"] [ONLY_ORIGIN="extend-1126-lr"] \
#   [ONLY_MULTIPLIERS="4.0"] \
#   [DEPEND=<job id>] [DRY=1] \
#       workflows/submit-molnl-batch-sweep.sh
#
# ARCH=gpt2 submits one job per arm. ARCH=bert submits a chain of segments per arm,
# because an arm of this grid outruns the 96-hour limit at xl and comes close at large.
#
# ONLY_BATCHES and ONLY_SIZES narrow what goes in, which is how a grid of 68 is started
# in parts without the inventory being edited. Unset, every arm of the architecture is
# submitted.
#
# ONLY_ORIGIN narrows by the inventory's origin column, and after 2026-11-26 the
# inventory holds arms that have already run: grid-1120 is the 68 of the original grid,
# extend-1126-lr the rates added above the cells whose top rate won, extend-1126-shape
# the one arm repeated at other block counts. Submitting without it would hand the
# launcher every finished arm again. A BERT segment of a finished run refuses in
# seconds rather than overwriting it, so that is noise rather than loss -- but it is
# 86 jobs of noise, and the record would no longer say what was started.

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

# The dry-run flag is DRY, and a near miss of that name is refused rather than ignored.
# On 2026-11-27 DRY_RUN=1 was typed in front of this script: the shell set a variable
# nothing reads, the submission went out as a real one, and twelve arms were started by
# a command meant to print them. Every other mistake here is recoverable by not running
# the output; this one is not. So anything that begins with DRY and is not DRY stops the
# script before a single sbatch is issued.
#
# The name is worth refusing loudly rather than aliasing, because the repository holds
# both. Counted over workflows/submit-*.sh: seventeen read DRY_RUN (fifteen rna, two
# genome) and six read DRY (three compounds, one each of genome, molnl, protein).
# Whichever name a reader carries over from the script they used last, one of the two
# is silent. Here it is not.
for _dry_like in $(compgen -v | grep '^DRY' || true); do
  [ "${_dry_like}" = "DRY" ] && continue
  echo "${_dry_like} is set, and this script reads only DRY." >&2
  echo "A dry run is DRY=1. Unset ${_dry_like} and say what was meant." >&2
  exit 1
done

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
only_o=""; [ -n "${ONLY_ORIGIN:-}" ] && only_o=" ${ONLY_ORIGIN} "
# ONLY_MULTIPLIERS narrows by the rate column's multiplier, which is the lever a
# staged launch needs: the top of the range answers whether the range is now wide
# enough, and the rates inside it are only worth running where it is not. The value
# is matched as the inventory writes it, to one decimal (0.7 1.0 1.4 2.0 2.8 4.0).
only_m=""; [ -n "${ONLY_MULTIPLIERS:-}" ] && only_m=" ${ONLY_MULTIPLIERS} "
# A misspelled origin would match nothing and submit nothing, which reads the same as
# "that part is already running". Names are checked against the ones the inventory uses.
KNOWN_ORIGINS=" grid-1120 extend-1126-lr extend-1126-shape "
for o in ${ONLY_ORIGIN:-}; do
  case "${KNOWN_ORIGINS}" in
    *" ${o} "*) ;;
    *) echo "ONLY_ORIGIN has ${o}, which is not one of${KNOWN_ORIGINS}" >&2; exit 1 ;;
  esac
done

# The fields are pulled out by name rather than by position, so a new column in the
# inventory does not shift what is read. awk rather than `read -a`: with IFS=tab a run
# of tabs -- which is what the empty cells of the other framework's columns are --
# collapses and shifts every later field.
rows=$(grep -v '^#' "${ARMS}" | awk -F'\t' -v want="${ARCH}" -v ob="${only_b}" -v os="${only_s}" -v oo="${only_o}" -v om="${only_m}" '
  NR == 1 { for (i = 1; i <= NF; i++) at[$i] = i;
            if (!("origin" in at)) { print "no origin column" > "/dev/stderr"; exit 1 }
            next }
  $at["arch"] == want \
    && (ob == "" || index(ob, " " $at["batch"] " ") > 0) \
    && (os == "" || index(os, " " $at["size"] " ") > 0) \
    && (oo == "" || index(oo, " " $at["origin"] " ") > 0) \
    && (om == "" || index(om, " " $at["lr_multiplier"] " ") > 0) {
    print $at["size"], $at["batch"], $at["lr_slug"], $at["config"], $at["origin"], $at["micro_batch"]
  }')
[ -n "${rows}" ] || {
  echo "no ${ARCH} arm in ${ARMS}${ONLY_BATCHES:+ at batch ${ONLY_BATCHES}}${ONLY_SIZES:+ at size ${ONLY_SIZES}}${ONLY_ORIGIN:+ origin ${ONLY_ORIGIN}}${ONLY_MULTIPLIERS:+ multiplier ${ONLY_MULTIPLIERS}}" >&2
  exit 1; }

submitted=0
while read -r size batch slug cfg origin micro; do
  [ -f "${cfg}" ] || { echo "no config at ${cfg}" >&2; exit 1; }
  # The §15 replicates share size, batch and rate with the arm they are compared
  # against, so without the block count in the name three jobs would carry one name and
  # the record would not say which ran.
  tag=""; [ "${origin}" = "extend-1126-shape" ] && tag="-m${micro}"
  if [ "${ARCH}" = "gpt2" ]; then
    name="mc-molnl-gpt2-b${batch}-${size}-lr${slug}${tag}"
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
    name="mc-molnl-bert-b${batch}-${size}-lr${slug}${tag}-s${seg}"
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
