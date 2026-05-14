#!/usr/bin/env bash
# Download the ChEBI-20 splits used by molecule captioning models.
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/chebi20/
#     train.txt validation.txt test.txt
#     manifest.json
#
# License: see https://github.com/blender-nlp/MolT5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init chebi20

BASE="https://raw.githubusercontent.com/blender-nlp/MolT5/main/ChEBI-20_data"
for split in train validation test; do
    ed_download "${BASE}/${split}.txt" "${split}.txt"
done

ed_finalize_manifest \
    "ChEBI-20 (MolT5 release)" \
    "https://github.com/blender-nlp/MolT5" \
    "see MolT5 license" \
    "$(date -u +%Y%m%d)"

cat <<'EOF'

Next step:
  ChEBI-20 ships TSV with columns (CID, SMILES, description); the
  loader accepts either .tsv, .csv, or .txt.  No further conversion is
  required.
EOF
