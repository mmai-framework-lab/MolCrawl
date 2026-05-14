#!/usr/bin/env bash
# Run every eval-data-*.sh downloader in turn.
#
# The script never aborts on per-task failures: missing credentials or
# network errors only mark that task as skipped / failed in the summary
# at the end.  This makes it safe to run on cron.
#
# Optional:
#   EVAL_DATA_TASKS  - space-separated subset of task slugs
#                      (default: all known tasks)
#   EVAL_DATA_DRY_RUN - print invocation list and exit

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_TASKS=(
    clinvar
    cosmic
    omim
    proteingym
    gnomad
    moleculenet
    moses
    chembl-heldout
    tape
    deeploc
    protein-foldability
    gue
    rna-benchmark
    tabula-sapiens
    replogle-perturb-seq
    molecule-nat-lang
    chebi20
    chemllmbench
)

if [ -n "${EVAL_DATA_TASKS:-}" ]; then
    read -r -a TASKS <<< "${EVAL_DATA_TASKS}"
else
    TASKS=("${DEFAULT_TASKS[@]}")
fi

if [ "${EVAL_DATA_DRY_RUN:-0}" = "1" ]; then
    echo "Tasks scheduled (dry-run):"
    for t in "${TASKS[@]}"; do
        echo "  ${SCRIPT_DIR}/eval-data-${t}.sh"
    done
    exit 0
fi

declare -a OK=()
declare -a SKIPPED=()
declare -a FAILED=()

for task in "${TASKS[@]}"; do
    script="${SCRIPT_DIR}/eval-data-${task}.sh"
    if [ ! -x "${script}" ]; then
        echo "[eval-data-all] missing or non-executable: ${script}" >&2
        FAILED+=("${task} (no script)")
        continue
    fi
    echo "============================================================"
    echo " ${task}"
    echo "============================================================"
    if "${script}"; then
        OK+=("${task}")
    else
        rc=$?
        if [ "${rc}" -eq 0 ]; then
            SKIPPED+=("${task}")
        else
            FAILED+=("${task} (exit ${rc})")
        fi
    fi
done

echo
echo "============================================================"
echo "Eval data download summary"
echo "============================================================"
printf 'OK      : %s\n' "${OK[*]:-(none)}"
printf 'SKIPPED : %s\n' "${SKIPPED[*]:-(none)}"
printf 'FAILED  : %s\n' "${FAILED[*]:-(none)}"

if [ "${#FAILED[@]}" -gt 0 ]; then
    exit 1
fi
