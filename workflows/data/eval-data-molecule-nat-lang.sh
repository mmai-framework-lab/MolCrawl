#!/usr/bin/env bash
# Register a molecule_nat_lang pair CSV.
#
# The pair file (smiles + caption) is produced by
# molcrawl/preparation/preparation_script_molecule_nat_lang.py.
# Provide its path through MOLECULE_NAT_LANG_SOURCE; this workflow
# only places it inside the standard eval directory and writes a
# manifest entry.
#
# Required:
#   MOLECULE_NAT_LANG_SOURCE - path to the prepared CSV
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/molecule_nat_lang/source.csv
#   manifest.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init molecule_nat_lang

if [ -z "${MOLECULE_NAT_LANG_SOURCE:-}" ]; then
    ed_skip_with_instructions \
        "set MOLECULE_NAT_LANG_SOURCE to the molecule_nat_lang pair CSV."
fi

dest="$(ed_dest)/source.csv"
cp "${MOLECULE_NAT_LANG_SOURCE}" "${dest}"
ed_register_existing "source.csv" "${MOLECULE_NAT_LANG_SOURCE}"

ed_finalize_manifest \
    "molecule_nat_lang pair CSV (project-internal)" \
    "https://github.com/deskull-m/MolCrawl-private" \
    "Internal" \
    "$(date -u +%Y%m%d)"
