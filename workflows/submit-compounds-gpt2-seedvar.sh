#!/bin/bash
# Submit the three compounds GPT-2 large runs that differ in nothing but the seed.
#
#   LEARNING_SOURCE_DIR=<input root> MODEL_OUTPUT_ROOT=<output root> \
#   SUBMIT_RECORD=<record file> [DRY=1] \
#       workflows/submit-compounds-gpt2-seedvar.sh
#
# They measure one spread: how far apart the end-of-run rise comes out when the
# condition is held fixed. The 2026-11-20 directive's §4 ruled out the schedule as the
# reason the existing 2,560 large arm rises +0.5260 from its minimum while the grid's
# 2,048 arm rises +0.0999 -- all four ladder runs decay over exactly their own length.
# What is left between the two is the batch and the rate that follows from it, and
# nothing measured so far says how much of a 5.3x difference in the rise is run-to-run
# spread. The grid's within-batch agreement is across sizes, not across seeds.
#
# Three runs, one condition: the ladder's own batch 2,560 and lr 1.6e-3 at large.
# Seed 42 is re-run rather than taken from the existing run, which is on commit
# 50c579f of 2026-10-01 -- a spread measured across code versions would carry the
# version as well as the seed. The existing run is left exactly as it is.
#
# One job each: a compounds GPT-2 large run of this length finishes inside the
# 96-hour limit (the grid's large arms took 3 h 58 m and 4 h 04 m for the same
# budget), so there is no segment chain and no continuation guard to think about.
#
# Each run goes through sbatch-watched.sh, so the id is recorded. DRY=1 prints what
# would be submitted.
set -uo pipefail

: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the packed compounds input root}"
: "${MODEL_OUTPUT_ROOT:?export MODEL_OUTPUT_ROOT to a runs root outside the input tree}"
: "${SUBMIT_RECORD:?export SUBMIT_RECORD to the file to append job ids to}"
DRY="${DRY:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

C=molcrawl/tasks/pretrain/configs/compounds
# sv42 is the control on today's code; seed1 and seed17 are the variance arms the
# 2026-09-30 directive's one exception allows.
SUFFIXES="sv42 seed1 seed17"

submitted=0
for suffix in ${SUFFIXES}; do
  cfg="${C}/gpt2_large_lr1p6e3_${suffix}.py"
  [ -f "${cfg}" ] || { echo "no config at ${cfg}" >&2; exit 1; }
  out="${MODEL_OUTPUT_ROOT}/compounds/gpt2-output/compounds-large-lr1p6e3-${suffix}"
  # A directory already there would be resumed from, not started afresh, and the
  # spread would be measured over a run that is partly another run.
  if [ -e "${out}" ]; then
    echo "output already exists: ${out}" >&2
    echo "move it aside first; this measurement needs three runs started from scratch" >&2
    exit 1
  fi
  name="mc-cmp-gpt2-large-lr1p6e3-${suffix}"
  args=(--job-name="${name}"
        --export=ALL,CONFIG="${cfg}",LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}",MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}"
        workflows/compounds-gpt2-grid.sbatch)
  if [ "${DRY}" = "1" ]; then
    echo "would submit ${name}  <- ${cfg}  -> ${out}"
  else
    jid=$(SUBMIT_RECORD="${SUBMIT_RECORD}" workflows/sbatch-watched.sh "${name}" -- "${args[@]}")
    echo "${jid}  ${name}"
  fi
  submitted=$((submitted + 1))
done

echo "${submitted} job(s) $([ "${DRY}" = "1" ] && echo 'would be submitted' || echo submitted)"
