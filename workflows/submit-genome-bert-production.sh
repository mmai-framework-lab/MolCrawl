#!/bin/bash
# Submit genome BERT production, one job per subset.
#
# Skips a subset that already carries .run_complete, and one that is already
# queued or running -- a duplicate submission once put two jobs on the same
# GPT-2 subset writing into one checkpoint tree.
#
# Every job stages 230G to node-local NVMe before it starts, so launching all
# twenty at once means a burst of concurrent reads on Lustre. That only delays
# the starts; once staged, the runs no longer touch it. BATCH throttles how many
# go in at a time if that burst needs limiting.
#
#   GENOME_SOURCE_ROOT=/path/to/learning_source ./workflows/submit-genome-bert-production.sh

set -uo pipefail
cd "$(dirname "$0")/.."

: "${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT to the learning_source root holding genome_sequence/}"
[ -d "${GENOME_SOURCE_ROOT}/genome_sequence" ] || {
    echo "GENOME_SOURCE_ROOT=${GENOME_SOURCE_ROOT} has no genome_sequence/ under it" >&2; exit 1; }

BATCH="${BATCH:-0}"          # 0 = no limit
DRY="${DRY:-0}"

# A subset is a directory that actually carries the BERT arrow tree. Name
# filtering let `gpt2-output` through once; this cannot.
mapfile -t SUBSETS < <(
    for d in "${GENOME_SOURCE_ROOT}/genome_sequence"/*/; do
        [ -d "${d}/training_ready_hf_dataset_bert" ] && basename "${d}"
    done | sort)
[ "${#SUBSETS[@]}" -gt 0 ] || { echo "no subsets found" >&2; exit 1; }

# SKIP is a comma-separated escape hatch for a subset already running under a
# job name this script would not recognise -- the in-flight check below only
# matches the mc-gen-bert-<subset> names it submits itself.
SKIP=",${SKIP:-},"
RUNNING=$(squeue -u "$USER" -h -o "%j" 2>/dev/null)
submitted=0 complete=0 inflight=0

for s in "${SUBSETS[@]}"; do
    if [ -f "learning_source_genome_runs/bert-small-${s}/.run_complete" ]; then
        complete=$((complete + 1)); continue
    fi
    if [[ "${SKIP}" == *",${s},"* ]]; then
        echo "  skipped by request: ${s}"; inflight=$((inflight + 1)); continue
    fi
    if grep -qx "mc-gen-bert-${s}" <<< "${RUNNING}"; then
        echo "  in flight, skipping: ${s}"; inflight=$((inflight + 1)); continue
    fi
    if [ "${BATCH}" -gt 0 ] && [ "${submitted}" -ge "${BATCH}" ]; then
        echo "  batch limit ${BATCH} reached, stopping"; break
    fi
    if [ "${DRY}" = "1" ]; then
        echo "  would submit: ${s}"
    else
        sbatch --job-name="mc-gen-bert-${s}" \
            --export=ALL,GENOME_SOURCE_ROOT="${GENOME_SOURCE_ROOT}",GENOME_SUBSET="${s}" \
            workflows/genome-bert-train.sbatch | sed "s/$/  ${s}/"
    fi
    submitted=$((submitted + 1))
done

echo "submitted=${submitted} already-complete=${complete} in-flight=${inflight} total=${#SUBSETS[@]}"
