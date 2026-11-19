#!/bin/bash
# Submit the compounds batch-size grid's arms, one job per arm, from the inventory.
#
#   LEARNING_SOURCE_DIR=<input root> MODEL_OUTPUT_ROOT=<output root> \
#   ARMS=<rt1-...arms.tsv> SUBMIT_RECORD=<record file> [ARCH=gpt2] [DRY=1] \
#       workflows/submit-compounds-batch-sweep.sh
#
# ARCH defaults to gpt2 and nothing else is submitted without being asked for by
# name. The reason is §8 of the 2026-11-19c directive, which states one GPU's column
# count twice and not the same way: its tables name BERT medium / large / xl 8 while
# its sentence says each size keeps its current value, and the adopted arms run at
# 160, 64 and 32. The BERT arms' micro-batch and accumulation therefore are not
# settled, and a run started under one reading cannot be compared with one started
# under the other. GPT-2 is unaffected -- its table and the current values agree, and
# both readings of §8 give GPT-2 the same four shapes -- so those 16 arms are
# submitted while that question is open.
#
# Each arm goes through sbatch-watched.sh, so the id is recorded and a failure
# recorder is chained behind it. DRY=1 prints what would be submitted.
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
  gpt2) ;;
  bert)
    echo "ARCH=bert: §8's two statements of one GPU's column count do not agree, so" >&2
    echo "the 40 BERT configs are not settled. Submit these only after that is" >&2
    echo "decided, and regenerate the configs if the decision is the other reading." >&2
    [ "${FORCE_BERT:-0}" = "1" ] || exit 1
    ;;
  *) echo "ARCH must be gpt2 or bert, got ${ARCH}" >&2; exit 1 ;;
esac

submitted=0
# The fields are pulled out by name rather than by position, so a new column in the
# inventory does not shift what is read. awk rather than `read -a`: IFS=tab treats a
# tab as whitespace, so a run of tabs -- which is what the empty cells of the other
# framework's columns are -- would collapse and shift every later field.
rows=$(grep -v '^#' "${ARMS}" | awk -F'\t' -v want="${ARCH}" '
  NR == 1 { for (i = 1; i <= NF; i++) at[$i] = i; next }
  $at["arch"] == want {
    print $at["size"], $at["batch"], $at["lr_slug"], $at["config"]
  }')
[ -n "${rows}" ] || { echo "no ${ARCH} arm in ${ARMS}" >&2; exit 1; }

while read -r size batch slug cfg; do
  [ -f "${cfg}" ] || { echo "no config at ${cfg}" >&2; exit 1; }
  name="mc-cmp-${ARCH}-${size}-b${batch}-lr${slug}"
  args=(--job-name="${name}"
        --export=ALL,CONFIG="${cfg}",LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}",MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}"
        "workflows/compounds-${ARCH}-grid.sbatch")
  if [ "${DRY}" = "1" ]; then
    echo "would submit ${name}  <- ${cfg}"
  else
    jid=$(SUBMIT_RECORD="${SUBMIT_RECORD}" workflows/sbatch-watched.sh "${name}" -- "${args[@]}")
    echo "${jid}  ${name}"
  fi
  submitted=$((submitted + 1))
done <<< "${rows}"

echo "${submitted} ${ARCH} arm(s) $([ "${DRY}" = "1" ] && echo 'would be submitted' || echo submitted)"
