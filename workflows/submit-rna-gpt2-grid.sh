#!/bin/bash
# Queue the whole rna GPT-2 grid, every link of every point, in one pass.
#
# Links within a point are joined with --dependency=afterany, so link N+1 starts
# once link N leaves the queue for any reason. The reason that matters is the
# 4-day wall limit: a job killed there ends TIMEOUT, which afterok would not
# follow. Nothing has to be running or watching for a chain to advance, so the
# grid carries across a holiday with nobody submitting.
#
# Each link resumes from out_dir on its own (init_from="resume" in the base
# config) and exits once iter_num reaches max_iters, so surplus links cost a
# startup each and nothing more. That is why the link count is the estimate plus
# one rather than the estimate: an s/step that comes in slow stalls the grid for
# five days otherwise.
#
# The first link of a point starts from scratch only because its out_dir is
# empty. GPT-2 has no "refuse to resume on link 1" check, so an out_dir with a
# checkpoint in it would be silently continued -- the caller is expected to have
# verified emptiness (scripts/rna_check_gpt2_grid.py does).
#
# Usage:
#   MODEL_OUTPUT_ROOT=... LEARNING_SOURCE_DIR=... EXPECT_COMMIT=...  \
#     ./workflows/submit-rna-gpt2-grid.sh
#
#   DRY_RUN=1 ...  prints the plan without submitting.
#   ONLY=xl        restricts to one size.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the rna learning_source root}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to a runs root outside any input tree}"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT to the account to charge}"

# size:links -- links from the measured wall clock of the earlier four runs,
# divided by 96 hours, rounded up, plus one. large and xl carry a second spare
# because their s/step was measured at a different micro-batch split.
POINTS=(
    "small:6e4:2"    "small:3e4:2"    "small:1p5e4:2"
    "medium:6e4:2"   "medium:3e4:2"   "medium:1p5e4:2"
    "large:6e4:3"    "large:3e4:3"    "large:1p5e4:3"
    "xl:6e4:4"       "xl:3e4:4"       "xl:1p5e4:4"      "xl:7p5e5:4"
)

# Longest first: the 4-day links are the ones that need to start today.
ORDER=(xl large medium small)

echo "=== rna GPT-2 grid ==="
echo "  MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT}"
echo "  LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR}"
echo "  EXPECT_COMMIT=${EXPECT_COMMIT:-<unset>}"
total_links=0
total_points=0

for want in "${ORDER[@]}"; do
    for entry in "${POINTS[@]}"; do
        IFS=: read -r size lr links <<< "${entry}"
        [ "${size}" = "${want}" ] || continue
        cfg="molcrawl/tasks/pretrain/configs/rna/gpt2_${size}_lr${lr}.py"
        [ -f "${cfg}" ] || { echo "missing config: ${cfg}" >&2; exit 1; }
        total_points=$((total_points + 1))

        dep=""
        for i in $(seq 1 "${links}"); do
            args=(--job-name="rna-gpt2-${size}-lr${lr}-L${i}"
                  --account="${SLURM_ACCOUNT}"
                  --export="ALL,GRID_CONFIG=${cfg}")
            [ -n "${dep}" ] && args+=(--dependency=afterany:"${dep}")

            if [ -n "${DRY_RUN:-}" ]; then
                echo "  [dry-run] ${size} lr${lr} link ${i}${dep:+ after ${dep}}"
                dep="<${size}-${lr}-L${i}>"
            else
                jid=$(sbatch --parsable "${args[@]}" workflows/rna-gpt2-grid.sbatch)
                echo "  ${size} lr${lr} link ${i}: job ${jid}${dep:+ (after ${dep})}"
                dep="${jid}"
            fi
            total_links=$((total_links + 1))
        done
    done
done
echo "=== ${total_points} points, ${total_links} links ==="
