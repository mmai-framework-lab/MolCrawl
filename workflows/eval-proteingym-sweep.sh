#!/usr/bin/env bash
# ProteinGym full-assay sweep.
#
# Runs the proteingym evaluator on every per-assay CSV under
#   ${LEARNING_SOURCE_DIR}/eval/proteingym/unpacked/DMS_ProteinGym_substitutions/
# (~217 assays for the canonical v1.3 release) against one representative
# (modality, arch, size) checkpoint. Each per-assay result lands in its
# own leaf so the dashboard can identify them individually; a follow-up
# aggregator can collapse the 217 into a single "mean Spearman" entry.
#
# Required:
#   MODEL_PATH       - protein decoder/encoder checkpoint
#   MODEL_SLUG       - <modality>-<arch>-<size> (output parent dir)
#                      e.g. ``protein_sequence-gpt2-medium``
#
# Optional:
#   ARCH             - default gpt2
#   MODALITY         - default protein_sequence
#   DEVICE           - cuda / cpu (default cuda)
#   BOOTSTRAP        - bootstrap CI resamples (default 100)
#   MAX_EXAMPLES     - per-assay variant cap (default 200)
#   PG_FILTER        - regex restricting assay CSVs by filename stem
#                      (e.g. ``PG_FILTER='^OPSD'`` for one assay only)
#   DRY_RUN          - "1" to print planned dispatches and exit

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_functions.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LSD="${LEARNING_SOURCE_DIR:-${REPO_ROOT}/../learning_source_20260316}"
export LEARNING_SOURCE_DIR="${LSD}"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${MODEL_SLUG:?MODEL_SLUG must be set (e.g. protein_sequence-gpt2-medium)}"

ARCH="${ARCH:-gpt2}"
MODALITY="${MODALITY:-protein_sequence}"
DEVICE="${DEVICE:-cuda}"
BOOTSTRAP="${BOOTSTRAP:-100}"
MAX="${MAX_EXAMPLES:-200}"
FILTER="${PG_FILTER:-}"
DRY="${DRY_RUN:-0}"

PG_DIR="${LSD}/eval/proteingym/unpacked/DMS_ProteinGym_substitutions"
if [[ ! -d "$PG_DIR" ]]; then
    echo "ERROR: ProteinGym DMS dir not found at $PG_DIR" >&2
    exit 1
fi

ASSAYS=$(ls "${PG_DIR}"/*.csv 2>/dev/null | sort)
TOTAL=$(echo "$ASSAYS" | wc -l)
echo "[pg-sweep] assays available: ${TOTAL}"
echo "[pg-sweep] model:            ${MODEL_PATH}"
echo "[pg-sweep] slug:             ${MODEL_SLUG}"
echo "[pg-sweep] device:           ${DEVICE}"

OK=0; SKIP=0; FAIL=0; PLANNED=0
for csv in $ASSAYS; do
    name=$(basename "$csv" .csv)
    if [[ -n "$FILTER" ]] && [[ ! "$name" =~ ${FILTER} ]]; then
        continue
    fi
    PLANNED=$((PLANNED+1))

    outdir="${LSD%/}/experiment_data/eval/${MODEL_SLUG}/proteingym_${name}_sweep"
    if find "$outdir" -name REPORT.md 2>/dev/null | grep -q .; then
        echo "[pg-sweep] SKIP done: ${name}"
        SKIP=$((SKIP+1))
        continue
    fi

    if [[ "$DRY" = "1" ]]; then
        echo "[pg-sweep] DRY:  ${name} -> ${outdir}"
        continue
    fi

    echo "[pg-sweep] RUN:  ${name}"
    mkdir -p "$outdir"
    OUTPUT_DIR="$outdir" RUNTAG="proteingym_${name}_sweep" \
        DEVICE="$DEVICE" BOOTSTRAP="$BOOTSTRAP" MAX_EXAMPLES="$MAX" \
        MODEL_PATH="$MODEL_PATH" PROTEINGYM_DATA="$csv" \
        ARCH="$ARCH" MODALITY="$MODALITY" \
        bash "${REPO_ROOT}/workflows/eval-proteingym.sh" \
            2>&1 | tail -3 || {
                echo "[pg-sweep] FAIL: ${name}" >&2
                FAIL=$((FAIL+1))
                continue
            }
    OK=$((OK+1))
done

echo
echo "[pg-sweep] DONE — planned=${PLANNED}, ok=${OK}, skipped=${SKIP}, failed=${FAIL}"
