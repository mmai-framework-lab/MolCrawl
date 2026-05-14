#!/usr/bin/env bash
# Download the MOSES reference SMILES split (ZINC-derived).
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/moses/
#     train.csv
#     test.csv
#     test_scaffolds.csv
#     manifest.json
#
# License: MIT (https://github.com/molecularsets/moses)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init moses

BASE="https://media.githubusercontent.com/media/molecularsets/moses/master/data"

ed_download "${BASE}/train.csv" "train.csv"
ed_download "${BASE}/test.csv" "test.csv"
ed_download "${BASE}/test_scaffolds.csv" "test_scaffolds.csv"

ed_finalize_manifest \
    "MOSES" \
    "https://github.com/molecularsets/moses" \
    "MIT" \
    "$(date -u +%Y%m%d)"
