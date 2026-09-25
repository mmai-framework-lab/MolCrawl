#!/bin/bash
# Queue the four 1.2e-3 points of rna GPT-2, every link of every point, in one pass.
#
# 6e-4 was the top of the grid and came out best at all four sizes, so what lies
# above it was never measured. These four sit one grid step above it. All four
# sizes go in together because the rate at which a run collapses falls as the
# model grows -- a result from one size does not carry to another -- and because
# waiting for one before submitting the next spends days of wall clock on nothing.
#
# Links within a point are joined with --dependency=afterany, so link N+1 starts
# once link N leaves the queue for any reason. The reason that matters is the
# 4-day wall limit: a job killed there ends TIMEOUT, which afterok would not
# follow. Nothing has to be running or watching for a chain to advance.
#
# Each link resumes from out_dir on its own and exits once iter_num reaches
# max_iters, so a surplus link costs one startup and nothing more. Link counts
# are the grid's submitted counts plus one.
#
# These runs measure on a fixed validation set (eval_val_fixed in the config),
# which the grid did not. Their eval numbers are not on the same footing as the
# grid's; compare within this set, or re-measure the grid the same way first.
#
# Usage:
#   MODEL_OUTPUT_ROOT=... LEARNING_SOURCE_DIR=... SLURM_ACCOUNT=... \
#     [EXPECT_COMMIT=...] ./workflows/submit-rna-gpt2-lr1p2e3.sh
#
#   DRY_RUN=1 ...  prints the plan without submitting.
#   ONLY=xl        restricts to one size.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the rna learning_source root}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to a runs root outside any input tree}"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT to the account to charge}"

# size:links -- the number of links the same size was submitted with in the grid,
# plus one. The grid measured small 190.3, medium 190.3, large 305.8 and xl 485.0
# GPU-h at 40,320 steps on four GPUs, so even at half that throughput every point
# finishes inside its links.
POINTS=(
    "small:3"
    "medium:3"
    "large:4"
    "xl:5"
)

# Longest first: the 4-day links are the ones that need to start today.
ORDER=(xl large medium small)
LR_TAG=1p2e3

echo "=== rna GPT-2 lr 1.2e-3 ==="
echo "  MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT}"
echo "  LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR}"
echo "  EXPECT_COMMIT=${EXPECT_COMMIT:-<unset>}"
total_links=0
total_points=0

for want in "${ORDER[@]}"; do
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${want}" ] && continue
    for entry in "${POINTS[@]}"; do
        IFS=: read -r size links <<< "${entry}"
        [ "${size}" = "${want}" ] || continue
        cfg="molcrawl/tasks/pretrain/configs/rna/gpt2_${size}_lr${LR_TAG}.py"
        [ -f "${cfg}" ] || { echo "missing config: ${cfg}" >&2; exit 1; }

        # Link 1 starts from scratch only because out_dir is empty. An out_dir
        # with a checkpoint in it would be continued silently, which would make
        # this point a continuation of something else.
        out="${MODEL_OUTPUT_ROOT}/rna-gpt2-${size}-lr${LR_TAG}"
        if compgen -G "${out}/ckpt*.pt" > /dev/null 2>&1; then
            echo "refusing: ${out} already holds a checkpoint" >&2; exit 1
        fi
        total_points=$((total_points + 1))

        dep=""
        for i in $(seq 1 "${links}"); do
            args=(--job-name="rna-gpt2-${size}-lr${LR_TAG}-L${i}"
                  --account="${SLURM_ACCOUNT}"
                  --export="ALL,GRID_CONFIG=${cfg}")
            [ -n "${dep}" ] && args+=(--dependency=afterany:"${dep}")

            if [ -n "${DRY_RUN:-}" ]; then
                echo "  [dry-run] ${size} lr${LR_TAG} link ${i}${dep:+ after ${dep}}"
                dep="<${size}-L${i}>"
            else
                jid=$(sbatch --parsable "${args[@]}" workflows/rna-gpt2-grid.sbatch)
                echo "  ${size} lr${LR_TAG} link ${i}: job ${jid}${dep:+ (after ${dep})}"
                dep="${jid}"
            fi
            total_links=$((total_links + 1))
        done
    done
done
echo "=== ${total_points} points, ${total_links} links ==="
