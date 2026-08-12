#!/bin/bash
# Submit a pre-linked chain of genome GPT-2 jobs for one subset.
#
# Links are connected with --dependency=afterany, so link N+1 starts once link N
# leaves the queue for any reason -- including a kill at the 4-day wall limit,
# which is the case that matters. Nothing has to be running or watching for the
# chain to advance, so it carries across a holiday unattended.
#
# Each link resumes from out_dir on its own and exits immediately once the run
# is complete, so over-provisioning links is safe: surplus links cost a startup
# each. Pick LINKS from the estimated wall-clock divided by the per-link limit,
# rounded up, plus one.
#
# Usage:
#   ./workflows/submit-genome-chain.sh <subset> [links] [micro_batch] [grad_accum]
#
# Example (3 links, micro-batch split 160/16):
#   ./workflows/submit-genome-chain.sh mammal_centered 3 160 16
#
# Dry run (print without submitting):
#   DRY_RUN=1 ./workflows/submit-genome-chain.sh mammal_centered 3

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUBSET="${1:?usage: submit-genome-chain.sh <subset> [links] [micro_batch] [grad_accum]}"
LINKS="${2:-3}"
MB="${3:-}"
GA="${4:-}"

EXPORT="ALL,GENOME_SUBSET=${SUBSET}"
[ -n "${MB}" ] && EXPORT="${EXPORT},MB=${MB}"
[ -n "${GA}" ] && EXPORT="${EXPORT},GA=${GA}"

if [ -n "${MB}" ] && [ -n "${GA}" ]; then
    product=$((MB * GA))
    [ "${product}" -eq 2560 ] || { echo "MB*GA must be 2560, got ${product}" >&2; exit 1; }
    [ $((GA % 4)) -eq 0 ] || { echo "grad_accum ${GA} must be divisible by world_size 4" >&2; exit 1; }
fi

echo "subset=${SUBSET} links=${LINKS} split=${MB:-config}/${GA:-config}"

dep=""
for i in $(seq 1 "${LINKS}"); do
    args=(--job-name="mc-gen-${SUBSET}-L${i}" --export="${EXPORT}")
    [ -n "${dep}" ] && args+=(--dependency=afterany:"${dep}")

    if [ -n "${DRY_RUN:-}" ]; then
        echo "  [dry-run] sbatch ${args[*]} workflows/genome-train.sbatch"
        dep="<link${i}>"
        continue
    fi

    jid=$(sbatch --parsable "${args[@]}" workflows/genome-train.sbatch)
    echo "  link ${i}: job ${jid}${dep:+ (after ${dep})}"
    dep="${jid}"
done
