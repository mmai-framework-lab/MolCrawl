#!/bin/bash
# Queue continuation links for rna GPT-2 nine-epoch points that are part way through.
#
# submit-rna-gpt2-9epoch.sh starts a point: it refuses an out_dir that already holds a
# checkpoint, because continuing one silently would make the point a continuation of
# whatever was there. That refusal is right for a first launch and wrong for the case
# here -- a maintenance window cancelled the running links, and the queued ones were
# consumed afterwards because EXPECT_COMMIT had been baked into their environment and
# the repository history was rewritten under them.
#
# So this script is the other half: it continues points that already hold a checkpoint
# and refuses the ones that do not, and it reads each run's own progress rather than
# being told. A run already at max_iters is skipped by name instead of being given a
# link that would start, find nothing to do and exit -- cheap, but it hides whether the
# count was right.
#
# Link counts are per size and passed in, because the right number depends on how far
# the run got before it stopped. Work it out from the steps remaining and the rate the
# run is actually holding, and add one spare; a surplus link exits in minutes.
#
# Usage:
#   LEARNING_SOURCE_DIR=... MODEL_OUTPUT_ROOT=... SLURM_ACCOUNT=... EXPECT_COMMIT=... \
#     POINTS="xl:4:7p5e5 1p5e4 3e4 6e4 1p2e3" ./workflows/submit-rna-gpt2-9epoch-continue.sh
#
#   POINTS entries are size:links:tags, one per argument-free word group, separated by
#   ';'. DRY_RUN=1 prints the plan without submitting.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the rna learning_source root}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to a runs root outside any input tree}"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT to the account to charge}"
: "${POINTS:?set POINTS to size:links:tags entries separated by ';'}"

echo "=== rna GPT-2 nine epochs, continuation links ==="
echo "  MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT}"
echo "  EXPECT_COMMIT=${EXPECT_COMMIT:-<unset>}"
total_jobs=0 total_points=0 skipped=0

_IFS="${IFS}"; IFS=';'; read -r -a _entries <<< "${POINTS}"; IFS="${_IFS}"
for entry in "${_entries[@]}"; do
    entry="${entry#"${entry%%[![:space:]]*}"}"
    [ -n "${entry}" ] || continue
    IFS=: read -r size links tags <<< "${entry}"
    for tag in ${tags}; do
        cfg="molcrawl/tasks/pretrain/configs/rna/gpt2_${size}_lr${tag}_9ep.py"
        [ -f "${cfg}" ] || { echo "missing config: ${cfg}" >&2; exit 1; }
        out="${MODEL_OUTPUT_ROOT}/rna-gpt2-${size}-lr${tag}-9ep"

        # The mirror image of the first-launch guard: this script continues, so an
        # out_dir without a checkpoint means the point never started and a link here
        # would quietly begin a fresh run under a name that claims otherwise.
        if ! compgen -G "${out}/ckpt*.pt" > /dev/null 2>&1; then
            echo "refusing: ${out} holds no checkpoint -- use submit-rna-gpt2-9epoch.sh" >&2
            exit 1
        fi

        _max=$(sed -n 's/^max_iters *= *\([0-9]*\).*/\1/p' "${cfg}" | tail -1)
        [ -n "${_max}" ] || { echo "cannot read max_iters from ${cfg}" >&2; exit 1; }
        _last=$(ls -d "${out}"/checkpoint-* 2>/dev/null \
                | sed 's/.*checkpoint-//' | sort -n | tail -1)
        _last="${_last:-0}"
        if [ "${_last}" -ge "${_max}" ]; then
            echo "  skip ${size} lr${tag}: already at ${_last}/${_max}"
            skipped=$((skipped + 1)); continue
        fi
        echo "  ${size} lr${tag}: at ${_last}/${_max}, ${links} link(s)"
        total_points=$((total_points + 1))

        dep=""
        for i in $(seq 1 "${links}"); do
            args=(--job-name="rna-gpt2-${size}-lr${tag}-9ep-C${i}"
                  --account="${SLURM_ACCOUNT}"
                  --export="ALL,GRID_CONFIG=${cfg}")
            [ -n "${dep}" ] && args+=(--dependency=afterany:"${dep}")
            if [ -n "${DRY_RUN:-}" ]; then
                echo "    [dry-run] link ${i}/${links}${dep:+ after ${dep}}"
                dep="<${size}-${tag}-C${i}>"
            elif [ -n "${SUBMIT_RECORD:-}" ]; then
                jid=$(workflows/sbatch-watched.sh "gpt2-9ep-${size}-${tag}-C${i}" -- \
                      "${args[@]}" workflows/rna-gpt2-grid.sbatch)
                echo "    link ${i}/${links}: job ${jid}${dep:+ (after ${dep})}"
                dep="${jid}"
            else
                jid=$(sbatch --parsable "${args[@]}" workflows/rna-gpt2-grid.sbatch)
                echo "    link ${i}/${links}: job ${jid}${dep:+ (after ${dep})}"
                dep="${jid}"
            fi
            total_jobs=$((total_jobs + 1))
        done
    done
done
echo "=== ${total_points} points, ${total_jobs} links, ${skipped} already complete ==="
