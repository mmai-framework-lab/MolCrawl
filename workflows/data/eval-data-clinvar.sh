#!/usr/bin/env bash
# Download ClinVar tab-delimited dumps from NCHI.
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/clinvar/
#     variant_summary.txt.gz
#     submission_summary.txt.gz
#     manifest.json
#
# License: Public domain (US Government).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init clinvar

ed_download \
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz" \
    "variant_summary.txt.gz"

ed_download \
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/submission_summary.txt.gz" \
    "submission_summary.txt.gz"

ed_finalize_manifest \
    "ClinVar (variant_summary, submission_summary)" \
    "https://www.ncbi.nlm.nih.gov/clinvar/" \
    "Public Domain (NCBI)" \
    "$(date -u +%Y%m%d)"

cat <<'EOF'

Next step:
  Convert variant_summary.txt.gz into the CSV expected by
  molcrawl.tasks.evaluation.clinvar via
  molcrawl/evaluation/gpt2/clinvar_data_preparation.py
  (the new task-centric loader expects reference_sequence /
   variant_sequence / ClinicalSignificance columns).
EOF
