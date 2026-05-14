#!/usr/bin/env bash
# Download the DeepLoc 2.0 Swissprot training+validation CSV.
#
# DTU's DeepLoc 2.0 page has a click-through landing for academic users
# (https://services.healthtech.dtu.dk/services/DeepLoc-2.0/), but the
# CSV itself is reachable via a direct URL with no auth. By running
# this downloader you confirm you have read and accepted the academic
# license at the DTU Health Tech site.
#
# Override DEEPLOC_URL if DTU restructures the URL or if you need a
# mirror. Override DEEPLOC_SOURCE to bypass the download entirely
# (e.g. when working from a pre-downloaded local copy).
#
# Required environment:
#   none (default URL is hard-coded)
# Optional:
#   DEEPLOC_URL     - direct URL of Swissprot_Train_Validation_dataset.csv
#   DEEPLOC_SOURCE  - path to a pre-downloaded CSV
#   DEEPLOC_NOTE    - free-form provenance note (origin URL etc.)
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/deeploc/source.csv  (raw multi-label CSV)
#   $LEARNING_SOURCE_DIR/eval/deeploc/deeploc.csv (single-label, ready for evaluator)
#   manifest.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init deeploc

dest_dir="$(ed_dest)"
DEEPLOC_URL="${DEEPLOC_URL:-https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/Swissprot_Train_Validation_dataset.csv}"
src_path="${DEEPLOC_SOURCE:-}"

if [ -n "${src_path}" ] && [ -f "${src_path}" ]; then
    if [ "${src_path}" != "${dest_dir}/source.csv" ]; then
        cp "${src_path}" "${dest_dir}/source.csv"
    fi
    ed_register_existing "source.csv" "${DEEPLOC_NOTE:-${DEEPLOC_URL}}"
else
    if ! ed_download "${DEEPLOC_URL}" "source.csv"; then
        ed_skip_with_instructions \
"failed to download DeepLoc 2.0 CSV from ${DEEPLOC_URL}.
Manual fallback:
  1. Visit https://services.healthtech.dtu.dk/services/DeepLoc-2.0/ and
     accept the academic license.
  2. Download Swissprot_Train_Validation_dataset.csv.
  3. Place the file at ${dest_dir}/source.csv (or set DEEPLOC_SOURCE=<path>) and re-run."
    fi
fi

# Reshape the multi-label CSV (10 binary columns) into a single-label
# CSV with columns: sequence, localisation, cluster_id, kingdom.
"$PYTHON" -m molcrawl.tasks.evaluation.deeploc.prepare_csv \
    --source-csv "${dest_dir}/source.csv" \
    --output-csv "${dest_dir}/deeploc.csv"

ed_finalize_manifest \
    "DeepLoc 2.0" \
    "https://services.healthtech.dtu.dk/services/DeepLoc-2.0/" \
    "Academic; see DTU SBC license" \
    "$(date -u +%Y%m%d)"
