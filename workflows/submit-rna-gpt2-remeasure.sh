#!/bin/bash
# Measure the GPT-2 one-GPU block counts again, with repeats and in a shuffled order.
#
# The 2026-11-20 probe ranked a 328M model ahead of a 104M one and put micro 16 and
# micro 64 three times ahead of micro 8 and micro 32 of the same size. All 31 arms did
# the same work -- 262,144 tokens an iteration, the right parameter count -- so the
# fast arms were real; what they were not is comparable. The arms that happened to run
# while the eight extension runs were quiet on the filesystem looked like fast shapes.
#
# Three things change here:
#   - every shape runs REPEATS times, so a shape is not judged on one time slot;
#   - the order is round-robin over shapes rather than grouped by size, so a busy hour
#     is spread across shapes instead of landing on one of them;
#   - the arm script reports the smallest iteration rather than the median, and the
#     submitter's reader takes the smallest across repeats. Reading can only add time.
#
# Usage:
#   LEARNING_SOURCE_DIR=... PROBE_OUT=... SLURM_ACCOUNT=... \
#     [REPEATS=3] [STEPS=300] [SUBMIT_RECORD=...] [DRY_RUN=1] \
#     ./workflows/submit-rna-gpt2-remeasure.sh
#
# One arm at a time, chained with afterany: two arms on one node warm the same files
# for each other, and a shape that does not fit is an answer that must not stop the
# chain.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the learning_source root}"
: "${PROBE_OUT:?set PROBE_OUT to a scratch root outside every input tree}"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT to the account to charge}"
REPEATS="${REPEATS:-3}"
STEPS="${STEPS:-300}"
GLOBAL=256
RUN_STEPS=1209600
SUBMIT_RECORD="${SUBMIT_RECORD:-${PROBE_OUT}/rna-gpt2-remeasure.tsv}"

# The same candidates as the first probe, so the two can be compared shape by shape.
# One per line: a continuation inside a quoted string leaves runs of spaces, and
# splitting those on a single space yields an empty field that becomes a 46th arm
# with no size.
SHAPES=$(printf '%s\n' small:8 small:16 small:32 small:64 \
                        medium:8 medium:16 medium:32 medium:64 \
                        large:4 large:8 large:16 large:32 \
                        xl:4 xl:8 xl:16)
SHAPE_COUNT=$(printf '%s\n' ${SHAPES} | grep -c .)

echo "=== rna GPT-2 block count, re-measured: ${REPEATS} repeats, global ${GLOBAL}," \
     "${STEPS} iters, commit $(git rev-parse --short HEAD)"
[ -n "${DRY_RUN:-}" ] || {
  mkdir -p "$(dirname "${SUBMIT_RECORD}")"
  [ -s "${SUBMIT_RECORD}" ] || printf 'submitted_at\tarch\tsize\tmicro\tglobal\taccum\tsteps\trepeat\tjob_id\tdepends_on\tcommit\n' > "${SUBMIT_RECORD}"
}

dep=""; n=0
for r in $(seq 1 "${REPEATS}"); do
  # Round-robin: repeat 1 runs the shapes in order, repeat 2 starts one shape further
  # along, and so on, so no shape keeps the same position in the queue.
  offset=$(( (r - 1) % SHAPE_COUNT ))
  # `sed -n '1,0p'` prints line 1 rather than nothing, so the first repeat would
  # carry its first shape twice -- 16 arms where 15 were asked for.
  if [ "${offset}" -eq 0 ]; then
    ordered=$(printf '%s\n' ${SHAPES})
  else
    ordered=$(printf '%s\n' ${SHAPES} | sed -n "$((offset + 1)),\$p"; \
              printf '%s\n' ${SHAPES} | sed -n "1,${offset}p")
  fi
  for shape in ${ordered}; do
    size="${shape%%:*}"; mb="${shape##*:}"
    [ $((GLOBAL % (mb * 4))) -eq 0 ] || { echo "global ${GLOBAL} does not divide by micro ${mb} x 4" >&2; exit 1; }
    name="mc-rna-tput2-gpt2-${size}-mb${mb}-r${r}"
    args=(--job-name="${name}" --account="${SLURM_ACCOUNT}" --export=ALL)
    [ -n "${dep}" ] && args+=(--dependency=afterany:"${dep}")
    if [ -n "${DRY_RUN:-}" ]; then
      echo "  [dry-run] ${size} micro ${mb} repeat ${r}${dep:+ after ${dep}}"
      dep="<${name}>"; n=$((n + 1)); continue
    fi
    jid=$(env SIZE="${size}" MB="${mb}" GLOBAL="${GLOBAL}" STEPS="${STEPS}" \
              REPEAT="${r}" PROBE_OUT="${PROBE_OUT}" RUN_STEPS="${RUN_STEPS}" \
              LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR}" \
              sbatch --parsable "${args[@]}" workflows/rna-gpt2-tput-arm.sbatch)
    echo "  ${size} micro ${mb} repeat ${r}: job ${jid}${dep:+ (after ${dep})}"
    printf '%s\tgpt2\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${size}" \
      "${mb}" "${GLOBAL}" "$((GLOBAL / mb))" "${STEPS}" "${r}" "${jid}" "${dep}" \
      "$(git rev-parse HEAD)" >> "${SUBMIT_RECORD}"
    dep="${jid}"; n=$((n + 1))
  done
done
echo "=== ${n} arms (${REPEATS} x ${SHAPE_COUNT} shapes), record ${SUBMIT_RECORD} ==="
