#!/usr/bin/env bash
# Register a scaffold-held-out ChEMBL CSV.
#
# Generation of the held-out CSV will eventually be wired into
# molcrawl/compounds/dataset/prepare_chembl.py via a
# --scaffold-split flag.  Until that lands, provide an existing CSV
# through CHEMBL_SCAFFOLD_HELDOUT_SOURCE.
#
# Required:
#   CHEMBL_SCAFFOLD_HELDOUT_SOURCE - path to the held-out CSV
# Optional:
#   CHEMBL_SCAFFOLD_TRAIN_SOURCE   - matching train CSV (encoder mode)
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/chembl_scaffold_heldout/{heldout,train}.csv
#   manifest.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init chembl_scaffold_heldout

if [ -z "${CHEMBL_SCAFFOLD_HELDOUT_SOURCE:-}" ]; then
    ed_skip_with_instructions \
        "set CHEMBL_SCAFFOLD_HELDOUT_SOURCE to the scaffold-held-out CSV."
fi

dest_dir="$(ed_dest)"
cp "${CHEMBL_SCAFFOLD_HELDOUT_SOURCE}" "${dest_dir}/heldout.csv"
ed_register_existing "heldout.csv" "${CHEMBL_SCAFFOLD_HELDOUT_SOURCE}"

if [ -n "${CHEMBL_SCAFFOLD_TRAIN_SOURCE:-}" ]; then
    cp "${CHEMBL_SCAFFOLD_TRAIN_SOURCE}" "${dest_dir}/train.csv"
    ed_register_existing "train.csv" "${CHEMBL_SCAFFOLD_TRAIN_SOURCE}"
fi

ed_finalize_manifest \
    "ChEMBL scaffold held-out (project-internal)" \
    "https://github.com/deskull-m/MolCrawl-private" \
    "Internal" \
    "$(date -u +%Y%m%d)"
