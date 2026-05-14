#!/usr/bin/env bash
# Download MoleculeNet CSVs (DeepChem mirror).
#
# Output (per task):
#   $LEARNING_SOURCE_DIR/eval/moleculenet/<task>/raw.csv
# Plus a single shared manifest.json listing every downloaded file.
#
# License: BSD-3-Clause (DeepChem) for the data wrappers; original
# datasets carry their individual licenses (Tox21 / SIDER etc.).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init moleculenet

BASE="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"

declare -A SOURCES=(
    [bbbp]="${BASE}/BBBP.csv"
    [tox21]="${BASE}/tox21.csv.gz"
    [toxcast]="${BASE}/toxcast_data.csv.gz"
    [sider]="${BASE}/sider.csv.gz"
    [clintox]="${BASE}/clintox.csv.gz"
    [bace]="${BASE}/bace.csv"
    [hiv]="${BASE}/HIV.csv"
    [muv]="${BASE}/muv.csv.gz"
    [esol]="${BASE}/delaney-processed.csv"
    [freesolv]="${BASE}/SAMPL.csv"
    [lipophilicity]="${BASE}/Lipophilicity.csv"
    [qm9_subset]="${BASE}/qm9.csv"
)

for task in "${!SOURCES[@]}"; do
    url="${SOURCES[$task]}"
    base_name="$(basename "${url}")"
    case "${base_name}" in
        *.gz) target="${task}/${base_name}";;
        *)    target="${task}/raw.csv";;
    esac
    ed_download "${url}" "${target}"
done

ed_finalize_manifest \
    "MoleculeNet (DeepChem mirror)" \
    "https://moleculenet.org/" \
    "BSD-3-Clause + per-dataset licenses" \
    "$(date -u +%Y%m%d)"

cat <<'EOF'

Next step:
  Decompress *.csv.gz files in-place, then rename / copy them to
  raw.csv inside each task directory before invoking
  workflows/eval-moleculenet.sh.
EOF
