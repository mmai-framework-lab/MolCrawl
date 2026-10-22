#!/bin/bash
# Queue the three lower BERT xl rates, and resume 5e-5 from where the wall cut it.
#
# rna-order-2026-10-14k §4 and §6. 5e-5 stopped at 7,200 steps with 113,760 to go;
# the three new points start from nothing. At the measured 24.75 s/it that is 32.6
# and 34.6 days, so ten and eleven links, chained with afterany.
#
# Usage:
#   MODEL_OUTPUT_ROOT=... LEARNING_SOURCE_DIR=... SLURM_ACCOUNT=... \
#     SUBMISSION_RECORD_DIR=... [EXPECT_COMMIT=...] ./workflows/submit-rna-bert-xl-lower.sh
#
#   DRY_RUN=1 ...   prints the plan without submitting.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the learning_source root holding rna/}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to a runs root outside any input tree}"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT to the account to charge}"
: "${SUBMISSION_RECORD_DIR:?set SUBMISSION_RECORD_DIR to where the submission record goes}"

# tag:links:first_segment -- 5e-5 resumes, so its first segment continues the chain
POINTS=("5e5:10:2" "2p5e5:11:1" "1p2e5:11:1" "6e6:11:1")

RECORD="${SUBMISSION_RECORD_DIR}/rna-bert-xl-lower-$(date -u +%Y%m%dT%H%M%S).tsv"
if [ -z "${DRY_RUN:-}" ]; then
    mkdir -p "${SUBMISSION_RECORD_DIR}"
    printf 'modality\tconfig\tsegment\tof\tjob\tdepends_on\n' > "${RECORD}"
fi
record() { [ -z "${DRY_RUN:-}" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "${RECORD}"; return 0; }

echo "=== rna BERT xl, lower rates ==="
echo "  EXPECT_COMMIT=${EXPECT_COMMIT:-<unset>}"
echo "  record: ${RECORD}"
total_jobs=0 total_points=0

for entry in "${POINTS[@]}"; do
    IFS=: read -r tag links first <<< "${entry}"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${tag}" ] && continue
    cfg="molcrawl/tasks/pretrain/configs/rna/bert_xl_lr${tag}.py"
    [ -f "${cfg}" ] || { echo "missing config: ${cfg}" >&2; exit 1; }
    out="${MODEL_OUTPUT_ROOT}/rna/bert-output/rna-xl-lr${tag}"

    # A point starting from scratch must not find a checkpoint; a point resuming must.
    if [ "${first}" = "1" ]; then
        if compgen -G "${out}/checkpoint-*" > /dev/null 2>&1; then
            echo "refusing: ${out} already holds a checkpoint" >&2; exit 1
        fi
    else
        compgen -G "${out}/checkpoint-*" > /dev/null 2>&1 || {
            echo "refusing: ${out} has no checkpoint to resume from" >&2; exit 1; }
    fi
    total_points=$((total_points + 1))

    prev=""
    last=$((first + links - 1))
    for seg in $(seq "${first}" "${last}"); do
        name="mc-bert-rna-xl-lr${tag}-s${seg}"
        args=(--job-name="${name}" --account="${SLURM_ACCOUNT}"
              --export="ALL,MODALITY=rna,CONFIG=${cfg},SEGMENT=${seg}")
        [ -n "${prev}" ] && args+=(--dependency=afterany:"${prev}")
        if [ -n "${DRY_RUN:-}" ]; then
            echo "  [dry-run] xl lr${tag} segment ${seg}${prev:+ after ${prev}}"
            job="<${name}>"
        else
            job=$(sbatch --parsable "${args[@]}" workflows/bert-grid.sbatch)
            echo "  xl lr${tag} segment ${seg}: job ${job}${prev:+ (after ${prev})}"
        fi
        record rna "${cfg}" "${seg}" "${last}" "${job}" "${prev:--}"
        prev="${job}"
        total_jobs=$((total_jobs + 1))
    done
done
echo "=== ${total_points} points, ${total_jobs} jobs ==="
