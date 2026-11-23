#!/bin/bash
# Submit the protein batch-size grid's arms from the derived inventory (2026-11-20
# directive). Mirrors workflows/submit-compounds-batch-sweep.sh.
#
#   LEARNING_SOURCE_DIR=<input root> MODEL_OUTPUT_ROOT=<output root> \
#   ARMS=<rt1-...arms.tsv> SUBMIT_RECORD=<record file> ARCH=bert \
#   [ONLY_BATCHES="256 512"] [DRY=1] \
#       workflows/submit-protein-batch-sweep.sh
#
# ARCH=bert submits a chain of afterany segments per arm (a protein BERT arm of this
# grid outruns the 96-hour limit at the larger sizes). ARCH=gpt2 is one job per arm
# (added once the GPT-2 micro-batch is measured; this submitter covers BERT for now).
#
# Segment counts come from the measured micro-32 step time (jobs 374634-374637) and the
# column budget. The budget is fixed in columns (85,708,800) and one GPU moves 32 x 1 =
# 32 columns per micro-step, 4 GPUs 128, so training is 85,708,800 / 128 = 669,600
# micro-steps per GPU, the same at every batch:
#
#   small  669,600 x 0.1668 s =  31.0 h  -> 1 + 1 spare
#   medium 669,600 x 0.4000 s =  74.4 h  -> 1 + 1 spare
#   large  669,600 x 0.6525 s = 121.4 h  -> 2 + 1 spare
#   xl     669,600 x 1.2168 s = 226.3 h  -> 3 + 1 spare   (batch-256 diagnostic, section 1.1)
#
# Those are per arm, independent of batch (fixed columns). What rides on the batch is
# the optimizer-step count (334,800 at batch 256 down to 20,925 at 4096) and the
# per-step overhead on it, which is not measured at this shape, so batch-256 arms get
# one further spare. A spare segment past the run's end is refused by the launcher in
# seconds.
#
# afterany, not afterok: a segment cut by the limit ends TIMEOUT; the launcher resumes
# only from TIMEOUT, so a crash stops the chain instead of resuming into it. The failure
# recorder (WATCH=1) goes on the last segment only, since every earlier one is meant to
# end TIMEOUT.
set -uo pipefail

: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the protein input root}"
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to a runs root outside the input tree}"
: "${ARMS:?set ARMS to the derived arm inventory TSV}"
: "${SUBMIT_RECORD:?export SUBMIT_RECORD to the file to append job ids to}"
ARCH="${ARCH:-bert}"
DRY="${DRY:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"
[ -f "${ARMS}" ] || { echo "no inventory at ${ARMS}" >&2; exit 1; }
case "${ARCH}" in
  bert|gpt2) ;;
  *) echo "ARCH must be bert or gpt2, got ${ARCH}" >&2; exit 1 ;;
esac

# Segment counts from the measured step time, wall = budget/(micro x 4) x step, which
# is the same at every batch. BERT: small 31h/medium 74h/large 121h/xl 226h. GPT-2 is
# faster (small 10h/medium 22h/large 45h/xl 83h), all under 96h but xl is close, so one
# real link plus one spare. Only BERT batch-256 arms (far more optimizer steps, overhead
# unmeasured) get an extra spare.
if [ "${ARCH}" = "bert" ]; then
  declare -A SEGMENTS=([small]=2 [medium]=2 [large]=3 [xl]=4)
  EXTRA_SEGMENT_BATCHES=" 256 "
else
  declare -A SEGMENTS=([small]=2 [medium]=2 [large]=2 [xl]=2)
  EXTRA_SEGMENT_BATCHES=" "
fi

only=""
[ -n "${ONLY_BATCHES:-}" ] && only=" ${ONLY_BATCHES} "
# Fields pulled by name (awk), so a new inventory column does not shift what is read,
# and tab-run collapse cannot shift fields past an empty cell.
rows=$(grep -v '^#' "${ARMS}" | awk -F'\t' -v want="${ARCH}" -v only="${only}" '
  NR == 1 { for (i = 1; i <= NF; i++) at[$i] = i; next }
  $at["arch"] == want && (only == "" || index(only, " " $at["batch"] " ") > 0) {
    print $at["size"], $at["batch"], $at["lr_slug"], $at["config"]
  }')
[ -n "${rows}" ] || { echo "no ${ARCH} arm in ${ARMS}${ONLY_BATCHES:+ at batch ${ONLY_BATCHES}}" >&2; exit 1; }

submitted=0
while read -r size batch slug cfg; do
  [ -f "${cfg}" ] || { echo "no config at ${cfg}" >&2; exit 1; }
  segments="${SEGMENTS[$size]}"
  case "${EXTRA_SEGMENT_BATCHES}" in *" ${batch} "*) segments=$((segments + 1)) ;; esac
  prev=""
  for seg in $(seq 1 "${segments}"); do
    name="mc-prot-${ARCH}-${size}-b${batch}-lr${slug}-s${seg}"
    dep=(); [ -n "${prev}" ] && dep=(--dependency="afterany:${prev}")
    if [ "${ARCH}" = "bert" ]; then
      args=(--job-name="${name}" "${dep[@]}"
            --export=ALL,MODALITY=protein_sequence,CONFIG="${cfg}",SEGMENT="${seg}",LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}",MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}"
            workflows/bert-grid.sbatch)
    else
      # GPT-2: nanoGPT resumes from out_dir itself (init_from=resume), so no SEGMENT arg;
      # the config carries out_dir and the rate. GRID_CONFIG, not CONFIG.
      args=(--job-name="${name}" "${dep[@]}"
            --export=ALL,GRID_CONFIG="${cfg}",LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}",MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}"
            workflows/protein-gpt2-grid.sbatch)
    fi
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
