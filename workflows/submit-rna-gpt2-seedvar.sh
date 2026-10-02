#!/bin/bash
# Queue the rna GPT-2 seed-variance points, every link of every point, up front.
#
# Two arms, 6e-4 and 1.2e-3, at seeds 1 and 17. With the grid's own 42 that gives
# three runs per arm, which is what a claimed ranking between rates needs behind it.
#
# Links chain with --dependency=afterany, since the 4-day wall ends a job as TIMEOUT
# and afterok would not follow it. Three links each: the grid's small arms finished
# inside one link (190.3 GPU-h, 47.6 hours on four GPUs), and three is the count the
# 1.2e-3 point was submitted with, which proved sufficient.
#
# Usage:
#   MODEL_OUTPUT_ROOT=... LEARNING_SOURCE_DIR=... SLURM_ACCOUNT=... \
#     [EXPECT_COMMIT=...] ./workflows/submit-rna-gpt2-seedvar.sh
#
#   DRY_RUN=1 ...   prints the plan without submitting.
#   ONLY=6e4        restricts to one rate.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${LEARNING_SOURCE_DIR:?set LEARNING_SOURCE_DIR to the rna learning_source root}"
: "${MODEL_OUTPUT_ROOT:?set MODEL_OUTPUT_ROOT to a runs root outside any input tree}"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT to the account to charge}"

RATES="${RATES:-6e4 1p2e3}"
SEEDS="${SEEDS:-1 17}"
LINKS="${LINKS:-3}"

echo "=== rna GPT-2 seed variance ==="
echo "  MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT}"
echo "  LEARNING_SOURCE_DIR=${LEARNING_SOURCE_DIR}"
echo "  EXPECT_COMMIT=${EXPECT_COMMIT:-<unset>}"
total_jobs=0
total_points=0

for rate in ${RATES}; do
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${rate}" ] && continue
    for sd in ${SEEDS}; do
        cfg="molcrawl/tasks/pretrain/configs/rna/gpt2_small_lr${rate}_seed${sd}.py"
        [ -f "${cfg}" ] || { echo "missing config: ${cfg}" >&2; exit 1; }

        # Link 1 starts from scratch only because out_dir is empty. An out_dir with a
        # checkpoint would be continued silently, making this point a continuation of
        # something else.
        out="${MODEL_OUTPUT_ROOT}/rna-gpt2-small-lr${rate}-seed${sd}"
        if compgen -G "${out}/ckpt*.pt" > /dev/null 2>&1; then
            echo "refusing: ${out} already holds a checkpoint" >&2; exit 1
        fi
        total_points=$((total_points + 1))

        dep=""
        for i in $(seq 1 "${LINKS}"); do
            args=(--job-name="rna-gpt2-small-lr${rate}-seed${sd}-L${i}"
                  --account="${SLURM_ACCOUNT}"
                  --export="ALL,GRID_CONFIG=${cfg}")
            [ -n "${dep}" ] && args+=(--dependency=afterany:"${dep}")

            if [ -n "${DRY_RUN:-}" ]; then
                echo "  [dry-run] lr${rate} seed${sd} link ${i}${dep:+ after ${dep}}"
                dep="<lr${rate}-seed${sd}-L${i}>"
            else
                jid=$(sbatch --parsable "${args[@]}" workflows/rna-gpt2-grid.sbatch)
                echo "  lr${rate} seed${sd} link ${i}: job ${jid}${dep:+ (after ${dep})}"
                dep="${jid}"
            fi
            total_jobs=$((total_jobs + 1))
        done
    done
done
echo "=== ${total_points} points, ${total_jobs} links ==="
