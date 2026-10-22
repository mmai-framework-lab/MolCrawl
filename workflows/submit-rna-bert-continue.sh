#!/bin/bash
# Queue continuation segments for rna BERT arms that an outside cancellation stopped.
#
# bert-grid.sbatch continues only from TIMEOUT unless RESUME_AFTER_CANCELLED names the
# predecessor, and a pending job's environment cannot be changed after submission. So a
# chain whose links were queued before the cancellation cannot be released into use: the
# first one refuses, and the rest refuse again for want of a segments.log entry. The
# chain has to be withdrawn and queued afresh with the variable set.
#
# RESUME_AFTER_CANCELLED authorises exactly one predecessor, so it goes on the first
# link only. Later links follow a TIMEOUT and need nothing; passing it to them anyway
# would be harmless -- the guard compares it against the predecessor's id -- but saying
# it once says what is meant.
#
# ARMS entries are tag:config:segment:links:cancelled_job, separated by ';':
#
#   large-lr1p7e4:molcrawl/.../bert_large_lr1p7e4.py:6:2:138658
#
# segment is the number of the segment to run next, which is one past the one that was
# cancelled; links is how many to chain from there. Work the count out from the steps
# remaining and the rate the arm is holding, and add one spare.
#
# Usage:
#   LEARNING_SOURCE_DIR=... MODEL_OUTPUT_ROOT=... SLURM_ACCOUNT=... MODALITY=rna \
#     ARMS="..." ./workflows/submit-rna-bert-continue.sh
#
#   DRY_RUN=1 ...   prints the plan without submitting.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the learning_source root}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to the root the grid outputs go under}"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT to the account to charge}"
: "${ARMS:?set ARMS to tag:config:segment:links:cancelled_job entries separated by ';'}"
MODALITY="${MODALITY:-rna}"

echo "=== rna BERT, continuation after an outside cancellation ==="
echo "  MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT}"
total_jobs=0 total_arms=0

_IFS="${IFS}"; IFS=';'; read -r -a _entries <<< "${ARMS}"; IFS="${_IFS}"
for entry in "${_entries[@]}"; do
    entry="${entry#"${entry%%[![:space:]]*}"}"
    [ -n "${entry}" ] || continue
    IFS=: read -r tag cfg seg links prev <<< "${entry}"
    [ -f "${cfg}" ] || { echo "missing config: ${cfg}" >&2; exit 1; }
    [ "${seg}" -ge 2 ] || { echo "${tag}: segment must be 2 or more here" >&2; exit 1; }

    # The guard checks the predecessor's end state itself, but it does so inside the
    # job. Checking here too means a wrong job id is a message now rather than a
    # refusal after the queue has turned over.
    _state=$(sacct -X -n -j "${prev}" -o State%20 2>/dev/null | head -1 | awk '{print $1}')
    [ "${_state}" = "CANCELLED" ] || {
        echo "${tag}: job ${prev} ended ${_state:-UNKNOWN}, not CANCELLED --" \
             "RESUME_AFTER_CANCELLED is for an outside cancellation only" >&2; exit 1; }

    echo "  ${tag}: segment ${seg}, ${links} link(s), after cancelled job ${prev}"
    total_arms=$((total_arms + 1))
    dep=""
    for i in $(seq 0 $((links - 1))); do
        s=$((seg + i))
        env="ALL,CONFIG=${cfg},SEGMENT=${s},MODALITY=${MODALITY}"
        # Only the first link follows the cancellation; the rest follow a TIMEOUT.
        [ "${i}" = "0" ] && env="${env},RESUME_AFTER_CANCELLED=${prev}"
        args=(--job-name="mc-bert-${MODALITY}-${tag}-s${s}"
              --account="${SLURM_ACCOUNT}" --export="${env}")
        [ -n "${dep}" ] && args+=(--dependency=afterany:"${dep}")
        if [ -n "${DRY_RUN:-}" ]; then
            echo "    [dry-run] segment ${s}${dep:+ after ${dep}}${i:+}"
            dep="<${tag}-s${s}>"
        elif [ -n "${SUBMIT_RECORD:-}" ]; then
            jid=$(workflows/sbatch-watched.sh "bert-${MODALITY}-${tag}-s${s}" -- \
                  "${args[@]}" workflows/bert-grid.sbatch)
            echo "    segment ${s}: job ${jid}${dep:+ (after ${dep})}"
            dep="${jid}"
        else
            jid=$(sbatch --parsable "${args[@]}" workflows/bert-grid.sbatch)
            echo "    segment ${s}: job ${jid}${dep:+ (after ${dep})}"
            dep="${jid}"
        fi
        total_jobs=$((total_jobs + 1))
    done
done
echo "=== ${total_arms} arms, ${total_jobs} segments ==="
