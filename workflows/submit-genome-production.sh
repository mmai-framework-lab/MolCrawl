#!/bin/bash
# Submit the genome subset production runs, one independent job per subset.
#
# Each subset now fits in a single job -- 10.7 h for the largest GPT-2 subset,
# 33.4 h for the largest BERT one, against the partition's 4-day limit -- so the
# runs are submitted flat rather than chained. Staging the split onto node-local
# NVMe is what buys that (8.2x for GPT-2); see genome-train.sbatch.
#
# There is no per-user concurrency limit and the partition has ample idle nodes,
# so submitting all 21 at once means the set finishes in roughly the wall-clock
# of its slowest member rather than the sum.
#
# Jobs are independent: one failing does not hold up the rest, and rerunning a
# subset resumes from its last checkpoint. A finished subset exits immediately
# on its completion marker, so re-running the whole sweep to pick up stragglers
# is cheap and safe.
#
# Usage:
#   ./workflows/submit-genome-production.sh <gpt2|bert> [subset ...]
#
#   # all 21 subsets
#   ./workflows/submit-genome-production.sh gpt2
#   # a subset of subsets
#   ./workflows/submit-genome-production.sh gpt2 mammal_centered global_random_seed4
#   # see what would be submitted
#   DRY_RUN=1 ./workflows/submit-genome-production.sh gpt2
#
# MB / GA override the micro-batch split (the product must stay 2560). A2 found
# the split makes little difference once the data is local, so leaving it to the
# config is fine.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="${1:?usage: submit-genome-production.sh <gpt2|bert> [subset ...]}"
shift || true
case "${MODEL}" in gpt2|bert) ;; *) echo "model must be gpt2 or bert" >&2; exit 1 ;; esac

# Check the corpus root here rather than letting each job discover it. The jobs
# do abort on their own, but by then 21 of them have been queued and will fail
# one after another; catching it once costs nothing and queues nothing.
: "${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT to the learning_source root holding genome_sequence/}"
[ -d "${GENOME_SOURCE_ROOT}/genome_sequence" ] || {
    echo "GENOME_SOURCE_ROOT=${GENOME_SOURCE_ROOT} has no genome_sequence/ under it" >&2; exit 1; }

ALL_SUBSETS=(
    mammal_centered
    eukaryote_matched_random_seed{1,2,3,4,5,6,7,8,9,10}
    global_random_seed{1,2,3,4,5,6,7,8,9,10}
)
SUBSETS=("$@")
[ ${#SUBSETS[@]} -eq 0 ] && SUBSETS=("${ALL_SUBSETS[@]}")

EXPORT_BASE="ALL,MODEL=${MODEL}"
[ -n "${MB:-}" ] && EXPORT_BASE="${EXPORT_BASE},MB=${MB}"
[ -n "${GA:-}" ] && EXPORT_BASE="${EXPORT_BASE},GA=${GA}"

if [ -n "${MB:-}" ] && [ -n "${GA:-}" ]; then
    [ $((MB * GA)) -eq 2560 ] || { echo "MB*GA must be 2560, got $((MB * GA))" >&2; exit 1; }
    # nanoGPT asserts grad_accum is divisible by world_size; HF does not.
    if [ "${MODEL}" = "gpt2" ] && [ $((GA % 4)) -ne 0 ]; then
        echo "gpt2: grad_accum ${GA} must be divisible by world_size 4" >&2; exit 1
    fi
fi

echo "model=${MODEL}  subsets=${#SUBSETS[@]}  split=${MB:-config}/${GA:-config}"

submitted=0
skipped=0
for subset in "${SUBSETS[@]}"; do
    out="learning_source_genome_runs/${MODEL}-small-${subset}"
    if [ -f "${out}/.run_complete" ]; then
        echo "  [done]  ${subset}  ($(cat "${out}/.run_complete"))"
        skipped=$((skipped + 1))
        continue
    fi

    args=(--job-name="mc-gen-${MODEL}-${subset}" --export="${EXPORT_BASE},GENOME_SUBSET=${subset}")
    if [ -n "${DRY_RUN:-}" ]; then
        echo "  [dry-run] sbatch ${args[*]} workflows/genome-train.sbatch"
        submitted=$((submitted + 1))
        continue
    fi

    jid=$(sbatch --parsable "${args[@]}" workflows/genome-train.sbatch)
    echo "  [job ${jid}] ${subset}"
    submitted=$((submitted + 1))
done

echo "submitted=${submitted} already-complete=${skipped}"
