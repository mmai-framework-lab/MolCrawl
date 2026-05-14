#!/usr/bin/env bash
# Download the ProteinGym substitution benchmark CSVs.
#
# ProteinGym moved distribution from marks.hms.harvard.edu to Zenodo
# (record 15293562, "ProteinGym") in 2025; the old Harvard URLs return
# 404. We fetch the DMS substitutions zip from Zenodo and keep the
# on-disk filename as ProteinGym_substitutions.zip so downstream
# consumers don't have to track the upstream rename.
# PROTEINGYM_INCLUDE_INDELS=1 adds the indels benchmark.
# PROTEINGYM_ZENODO_ID overrides the Zenodo record id for future
# refreshes.
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/proteingym/
#     ProteinGym_substitutions.zip
#     manifest.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init proteingym

ZENODO_BASE="https://zenodo.org/records/${PROTEINGYM_ZENODO_ID:-15293562}/files"

ed_download \
    "${ZENODO_BASE}/DMS_ProteinGym_substitutions.zip" \
    "ProteinGym_substitutions.zip"

if [ "${PROTEINGYM_INCLUDE_INDELS:-0}" = "1" ]; then
    ed_download \
        "${ZENODO_BASE}/DMS_ProteinGym_indels.zip" \
        "ProteinGym_indels.zip"
fi

ed_finalize_manifest \
    "ProteinGym" \
    "https://proteingym.org/" \
    "CC-BY-4.0" \
    "$(date -u +%Y%m%d)"

cat <<'EOF'

Next step:
  unzip ProteinGym_substitutions.zip into the same directory before
  running molcrawl.tasks.evaluation.proteingym.
EOF
