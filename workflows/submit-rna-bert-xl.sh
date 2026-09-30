#!/bin/bash
# Queue the three rna BERT xl points, every segment up front.
#
# rna had no xl. large collapses between 2.3e-4 and 3e-4, and in rna the boundary
# falls by about 1.8 per step of size, which puts xl near 1.4e-4. The three points
# are 1e-4 / 1.4e-4 / 2e-4, a step of 1.4, so wherever the boundary is inside that
# span it is located to within one step -- and if 2e-4 survives or 1e-4 collapses,
# that says the span itself was placed wrong, which is also an answer.
#
# Segments are chained with --dependency=afterany, so segment N+1 starts once N
# leaves the queue for any reason. The reason that matters is the 4-day wall: a job
# killed there ends TIMEOUT, which afterok would not follow.
#
# Segment count. A probe at 8 x 80 measured 28.53 s/it steady state, which is 12,113
# steps in four days, so 120,960 needs ten segments; one spare makes eleven. The
# probe ran with 38 jobs co-resident across the modalities, heavier than this will
# see once the re-measurement jobs drain, so the rate is a floor rather than a best
# case and no further margin is added. A surplus segment finds max_steps reached and
# exits in minutes.
#
# Usage:
#   MODEL_OUTPUT_ROOT=... LEARNING_SOURCE_DIR=... SLURM_ACCOUNT=... \
#     SUBMISSION_RECORD_DIR=... [EXPECT_COMMIT=...] ./workflows/submit-rna-bert-xl.sh
#
#   DRY_RUN=1 ...   prints the plan without submitting.
#   ONLY=1p4e4      restricts to one rate.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the learning_source root holding rna/}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to a runs root outside any input tree}"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT to the account to charge}"
: "${SUBMISSION_RECORD_DIR:?set SUBMISSION_RECORD_DIR to where the submission record goes}"

TAGS="${TAGS:-1e4 1p4e4 2e4}"
SEGMENTS="${SEGMENTS:-11}"

RECORD="${SUBMISSION_RECORD_DIR}/rna-bert-xl-$(date -u +%Y%m%dT%H%M%S).tsv"
if [ -z "${DRY_RUN:-}" ]; then
    mkdir -p "${SUBMISSION_RECORD_DIR}"
    printf 'modality\tconfig\tsegment\tof\tjob\tdepends_on\n' > "${RECORD}"
fi
record() { [ -z "${DRY_RUN:-}" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "${RECORD}"; return 0; }

echo "=== rna BERT xl ==="
echo "  MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT}"
echo "  LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR}"
echo "  EXPECT_COMMIT=${EXPECT_COMMIT:-<unset>}"
echo "  record: ${RECORD}"
total_jobs=0
total_points=0

for tag in ${TAGS}; do
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${tag}" ] && continue
    cfg="molcrawl/tasks/pretrain/configs/rna/bert_xl_lr${tag}.py"
    [ -f "${cfg}" ] || { echo "missing config: ${cfg}" >&2; exit 1; }

    # Segment 1 starts from scratch only because model_path is empty. A directory
    # with a checkpoint would be resumed silently, making this point a continuation
    # of something else.
    out="${MODEL_OUTPUT_ROOT}/rna/bert-output/rna-xl-lr${tag}"
    if compgen -G "${out}/checkpoint-*" > /dev/null 2>&1; then
        echo "refusing: ${out} already holds a checkpoint" >&2; exit 1
    fi
    total_points=$((total_points + 1))

    prev=""
    for seg in $(seq 1 "${SEGMENTS}"); do
        name="mc-bert-rna-xl-lr${tag}-s${seg}"
        args=(--job-name="${name}" --account="${SLURM_ACCOUNT}"
              --export="ALL,MODALITY=rna,CONFIG=${cfg},SEGMENT=${seg}")
        [ -n "${prev}" ] && args+=(--dependency=afterany:"${prev}")

        if [ -n "${DRY_RUN:-}" ]; then
            echo "  [dry-run] xl lr${tag} segment ${seg}/${SEGMENTS}${prev:+ after ${prev}}"
            job="<${name}>"
        else
            job=$(sbatch --parsable "${args[@]}" workflows/bert-grid.sbatch)
            echo "  xl lr${tag} segment ${seg}/${SEGMENTS}: job ${job}${prev:+ (after ${prev})}"
        fi
        record rna "${cfg}" "${seg}" "${SEGMENTS}" "${job}" "${prev:--}"
        prev="${job}"
        total_jobs=$((total_jobs + 1))
    done
done
echo "=== ${total_points} points, ${total_jobs} jobs ==="
