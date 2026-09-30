#!/bin/bash
# Submit the 13 compounds BERT runs of compounds-order-2026-09-30: the size grid of
# §4 (medium / large / xl at three rates each) and the seed replicates of §3
# (seeds 1 and 17 on the 5e-4 and 1e-3 arms), with every segment queued up front and
# chained by --dependency=afterany.
#
# Nothing is submitted unless SUBMIT=1. Without it this prints each sbatch command it
# would run, with placeholder ids for the dependencies.
#
#   MAIN=<main checkout> bash workflows/submit-compounds-bert-grid.sh           # print
#   MAIN=<main checkout> SUBMIT=1 bash workflows/submit-compounds-bert-grid.sh  # submit
#
# Run it from the checkout whose code the runs should use: SLURM_SUBMIT_DIR is where
# every segment reads its code, for as long as the grid takes.
#
# Segments come from the measured bf16 step time of each size (jobs 150891-150894, the
# shape each config now carries) times 15,000 steps, plus the measured cost of 150
# evaluation points, divided by the 96-hour limit and rounded up, plus one spare. A
# segment beyond the end of a run is refused by the launcher in seconds, so a spare
# costs a few seconds of allocation and covers a slower node.
#
#   small  3.13 s/step + 0.26 h eval = 13.3 h -> 1 + 1
#   medium 7.21 s/step + 0.71 h eval = 30.8 h -> 1 + 1
#   large 12.14 s/step + 0.97 h eval = 51.6 h -> 1 + 1
#   xl    23.70 s/step + 1.63 h eval = 100.4 h -> 2 + 1
#
# afterany, not afterok: a segment cut by the time limit ends TIMEOUT, and afterok
# would never start the next one. The launcher continues only from TIMEOUT, so a
# segment that crashed stops the chain instead of resuming into the same failure.
set -euo pipefail
: "${MAIN:?MAIN=<main checkout> (learning_source trees, miniconda)}"
SUBMIT="${SUBMIT:-0}"
RECORD="${RECORD:-${MAIN}/learning_source_bert_grid_submissions/compounds-$(date +%Y%m%dT%H%M%S).tsv}"
C=molcrawl/tasks/pretrain/configs/compounds
LSD="${MAIN}/learning_source_20260805_compounds_packed"
OUTROOT="${MAIN}/learning_source_compounds_runs"

declare -A SEGMENTS=([small]=2 [medium]=2 [large]=2 [xl]=3)

sb() {
  if [ "${SUBMIT}" = "1" ]; then sbatch --parsable "$@"
  else
    local a name=""
    for a in "$@"; do case "${a}" in --job-name=*) name="${a#--job-name=}";; esac; done
    echo "DRYRUN sbatch $*" >&2; echo "<${name}>"
  fi
}

if [ "${SUBMIT}" = "1" ]; then
  mkdir -p "$(dirname "${RECORD}")"
  printf 'config\tsegment\tof\tjob\tdepends_on\n' > "${RECORD}"
fi
record() { [ "${SUBMIT}" = "1" ] && printf '%s\t%s\t%s\t%s\t%s\n' "$@" >> "${RECORD}"; return 0; }

arm() {  # size config-basename job-name-suffix
  local size="$1" base="$2" suffix="$3"
  local k="${SEGMENTS[$size]}" cfg="${C}/${base}.py" prev="" seg name job dep
  [ -f "${cfg}" ] || { echo "no config at ${cfg}" >&2; exit 1; }
  for seg in $(seq 1 "${k}"); do
    name="mc-bert-compounds-${suffix}-s${seg}"
    dep=(); [ -n "${prev}" ] && dep=(--dependency="afterany:${prev}")
    job=$(sb --job-name="${name}" "${dep[@]}" \
          --export="ALL,MODALITY=compounds,LEARNING_SOURCE_DIR=${LSD},MODEL_OUTPUT_ROOT=${OUTROOT},CONFIG=${cfg},SEGMENT=${seg}" \
          workflows/bert-grid.sbatch)
    record "${cfg}" "${seg}" "${k}" "${job}" "${prev:--}"
    echo "${base} segment ${seg}/${k} -> ${job}${prev:+ (after ${prev})}"
    prev="${job}"
  done
}

# Longest first, so the runs that decide when the grid finishes are queued first.
for tag in 8e5 1p5e4 2p8e4; do arm xl "bert_xl_lr${tag}" "xl-lr${tag}"; done
for tag in 1p5e4 2p8e4 5e4; do arm large "bert_large_lr${tag}" "large-lr${tag}"; done
for tag in 2p8e4 5e4 1e3; do arm medium "bert_medium_lr${tag}" "medium-lr${tag}"; done
for lr in 5e4 1e3; do
  for s in 1 17; do arm small "bert_small_lr${lr}_seed${s}" "small-lr${lr}-seed${s}"; done
done

echo
if [ "${SUBMIT}" = "1" ]; then echo "record: ${RECORD}"; else echo "DRY RUN -- nothing submitted. Re-run with SUBMIT=1."; fi
