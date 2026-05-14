#!/usr/bin/env bash
# Download OMIM disease/gene tables.  OMIM requires a free API key
# obtained at https://www.omim.org/api and the user must accept the
# terms of use.  The API key must be provided through OMIM_API_KEY.
#
# Easiest setup: copy ``.env.example`` to ``.env`` at the repo root,
# fill in the OMIM_API_KEY line, and re-run this workflow.  ``.env`` is
# gitignored and auto-sourced by ``workflows/common_functions.sh`` so
# OMIM_API_KEY gets exported automatically.  Approval typically takes
# ~1 business day after registration.
#
# Required environment:
#   OMIM_API_KEY    - personal OMIM API key
#
# Optional:
#   OMIM_FILES      - space-separated list of files to download
#                     (default: mim2gene.txt morbidmap.txt genemap2.txt)
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/omim/
#     <selected files>
#     manifest.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init omim

if [ -z "${OMIM_API_KEY:-}" ]; then
    ed_skip_with_instructions \
        "set OMIM_API_KEY to download OMIM tables (https://www.omim.org/api)."
fi

FILES="${OMIM_FILES:-mim2gene.txt morbidmap.txt genemap2.txt}"

for filename in ${FILES}; do
    url="https://data.omim.org/downloads/${OMIM_API_KEY}/${filename}"
    ed_download "${url}" "${filename}"
done

ed_finalize_manifest \
    "OMIM (mim2gene / morbidmap / genemap2)" \
    "https://www.omim.org/" \
    "Restricted; see OMIM Terms of Use" \
    "$(date -u +%Y%m%d)"
