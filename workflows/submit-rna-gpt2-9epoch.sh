#!/bin/bash
# Queue the seventeen GPT-2 points at nine epochs, every link up front.
#
# Links chain with --dependency=afterany: the 4-day wall ends a job as TIMEOUT and
# afterok would not follow it. Counts come from the 3-epoch runs' measured wall
# clock, tripled for the longer schedule, plus one spare. A surplus link finds
# iter_num at max_iters and exits in minutes.
#
#   small / medium  47.6 h at 40,320  ->  142.7 h  ->  2 links + 1
#   large           76.5 h            ->  229.4 h  ->  3 links + 1
#   xl             121.2 h            ->  363.8 h  ->  4 links + 1
#
# Usage:
#   MODEL_OUTPUT_ROOT=... LEARNING_SOURCE_DIR=... SLURM_ACCOUNT=... \
#     [EXPECT_COMMIT=...] ./workflows/submit-rna-gpt2-9epoch.sh
#
#   DRY_RUN=1 ...   prints the plan without submitting.
#   ONLY=xl         restricts to one size.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the rna learning_source root}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to a runs root outside any input tree}"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT to the account to charge}"

# size:links:rate tags
POINTS=(
    "small:3:1p5e4 3e4 6e4 1p2e3"
    "medium:3:1p5e4 3e4 6e4 1p2e3"
    "large:4:1p5e4 3e4 6e4 1p2e3"
    "xl:5:7p5e5 1p5e4 3e4 6e4 1p2e3"
)
ORDER=(xl large medium small)   # longest first

echo "=== rna GPT-2, nine epochs ==="
echo "  MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT}"
echo "  EXPECT_COMMIT=${EXPECT_COMMIT:-<unset>}"
total_jobs=0 total_points=0

for want in "${ORDER[@]}"; do
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${want}" ] && continue
    for entry in "${POINTS[@]}"; do
        IFS=: read -r size links tags <<< "${entry}"
        [ "${size}" = "${want}" ] || continue
        for tag in ${tags}; do
            cfg="molcrawl/tasks/pretrain/configs/rna/gpt2_${size}_lr${tag}_9ep.py"
            [ -f "${cfg}" ] || { echo "missing config: ${cfg}" >&2; exit 1; }

            # Link 1 starts from scratch only because out_dir is empty. An out_dir
            # holding a checkpoint would be continued silently, making this point a
            # continuation of something else -- here, of the 3-epoch run.
            out="${MODEL_OUTPUT_ROOT}/rna-gpt2-${size}-lr${tag}-9ep"
            if compgen -G "${out}/ckpt*.pt" > /dev/null 2>&1; then
                echo "refusing: ${out} already holds a checkpoint" >&2; exit 1
            fi
            total_points=$((total_points + 1))

            dep=""
            for i in $(seq 1 "${links}"); do
                args=(--job-name="rna-gpt2-${size}-lr${tag}-9ep-L${i}"
                      --account="${SLURM_ACCOUNT}"
                      --export="ALL,GRID_CONFIG=${cfg}")
                [ -n "${dep}" ] && args+=(--dependency=afterany:"${dep}")
                if [ -n "${DRY_RUN:-}" ]; then
                    echo "  [dry-run] ${size} lr${tag} link ${i}/${links}${dep:+ after ${dep}}"
                    dep="<${size}-${tag}-L${i}>"
                else
                    jid=$(sbatch --parsable "${args[@]}" workflows/rna-gpt2-grid.sbatch)
                    echo "  ${size} lr${tag} link ${i}/${links}: job ${jid}${dep:+ (after ${dep})}"
                    dep="${jid}"
                fi
                total_jobs=$((total_jobs + 1))
            done
        done
    done
done
echo "=== ${total_points} points, ${total_jobs} links ==="
