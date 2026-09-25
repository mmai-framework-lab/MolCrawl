#!/bin/bash
# Queue the nine points that fill in the rna BERT grid, every segment up front.
#
# The first grid left each size with its best rate at an end of the range:
# small and medium were best at 3e-4 and collapsed at 1e-3; large was best at
# 1e-4 and collapsed at 3e-4. A best at an end is not bracketed. These nine
# points divide each open span into four in the logarithm and take the three
# interior marks, so wherever the collapse begins it is located to within one
# step of 1.35x (small, medium) or 1.32x (large).
#
# All nine go in together. Waiting for one before submitting the next spends
# days of wall clock doing nothing, and a rate that collapses declares itself
# early -- every one of the four collapses in the first grid happened by step
# 24,500, inside the first segment.
#
# Segments are chained with --dependency=afterany, so segment N+1 starts once
# segment N leaves the queue for any reason. The reason that matters is the
# 4-day wall: a job killed there ends TIMEOUT, which afterok would not follow.
#
# Segment counts. With six runs sharing the filesystem, four days carried small
# 82,100 steps, medium 46,200 and large 29,100. These nine, the three surviving
# points of the first grid and the four GPT-2 points make sixteen runs on the
# same storage, and a thirteen-run episode earlier ran at less than half the
# throughput. The counts below assume half, plus one spare. A surplus segment
# finds max_steps reached and exits in minutes, so over-provisioning costs
# startups, not GPU-hours.
#
# Usage:
#   MODEL_OUTPUT_ROOT=... LEARNING_SOURCE_DIR=... SLURM_ACCOUNT=... \
#     SUBMISSION_RECORD_DIR=... [EXPECT_COMMIT=...] \
#     ./workflows/submit-rna-bert-grid-fill.sh
#
#   DRY_RUN=1 ...     prints the plan without submitting.
#   ONLY=large        restricts to one size.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the learning_source root holding rna/}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to a runs root outside any input tree}"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT to the account to charge}"
: "${SUBMISSION_RECORD_DIR:?set SUBMISSION_RECORD_DIR to where the submission record goes}"

# size:segments:rate tags
POINTS=(
    "small:4:4p1e4 5p5e4 7p4e4"
    "medium:7:4p1e4 5p5e4 7p4e4"
    "large:10:1p3e4 1p7e4 2p3e4"
)

# Longest first: large needs the most wall clock, so it should start today.
ORDER=(large medium small)

RECORD="${SUBMISSION_RECORD_DIR}/rna-bert-fill-$(date -u +%Y%m%dT%H%M%S).tsv"
if [ -z "${DRY_RUN:-}" ]; then
    mkdir -p "${SUBMISSION_RECORD_DIR}"
    printf 'modality\tconfig\tsegment\tof\tjob\tdepends_on\n' > "${RECORD}"
fi
record() { [ -z "${DRY_RUN:-}" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "${RECORD}"; return 0; }

echo "=== rna BERT grid fill ==="
echo "  MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT}"
echo "  LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR}"
echo "  EXPECT_COMMIT=${EXPECT_COMMIT:-<unset>}"
echo "  record: ${RECORD}"
total_jobs=0
total_points=0

for want in "${ORDER[@]}"; do
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${want}" ] && continue
    for entry in "${POINTS[@]}"; do
        IFS=: read -r size segs tags <<< "${entry}"
        [ "${size}" = "${want}" ] || continue
        for tag in ${tags}; do
            cfg="molcrawl/tasks/pretrain/configs/rna/bert_${size}_lr${tag}.py"
            [ -f "${cfg}" ] || { echo "missing config: ${cfg}" >&2; exit 1; }

            # Segment 1 starts from scratch only because model_path is empty.
            # A directory with a checkpoint would be resumed silently, making
            # this point a continuation of something else.
            out="${MODEL_OUTPUT_ROOT}/rna/bert-output/rna-${size}-lr${tag}"
            if compgen -G "${out}/checkpoint-*" > /dev/null 2>&1; then
                echo "refusing: ${out} already holds a checkpoint" >&2; exit 1
            fi
            total_points=$((total_points + 1))

            prev=""
            for seg in $(seq 1 "${segs}"); do
                name="mc-bert-rna-${size}-lr${tag}-s${seg}"
                args=(--job-name="${name}" --account="${SLURM_ACCOUNT}"
                      --export="ALL,MODALITY=rna,CONFIG=${cfg},SEGMENT=${seg}")
                [ -n "${prev}" ] && args+=(--dependency=afterany:"${prev}")

                if [ -n "${DRY_RUN:-}" ]; then
                    echo "  [dry-run] ${size} lr${tag} segment ${seg}/${segs}${prev:+ after ${prev}}"
                    job="<${name}>"
                else
                    job=$(sbatch --parsable "${args[@]}" workflows/bert-grid.sbatch)
                    echo "  ${size} lr${tag} segment ${seg}/${segs}: job ${job}${prev:+ (after ${prev})}"
                fi
                record rna "${cfg}" "${seg}" "${segs}" "${job}" "${prev:--}"
                prev="${job}"
                total_jobs=$((total_jobs + 1))
            done
        done
    done
done
echo "=== ${total_points} points, ${total_jobs} jobs ==="
