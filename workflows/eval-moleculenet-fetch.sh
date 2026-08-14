#!/usr/bin/env bash
# Fetch the MoleculeNet CSVs the property-prediction evaluator reads.
#
# molcrawl/tasks/evaluation/moleculenet/data_preparation.py expects, per sub-task,
#   $MOLECULENET_DIR/<task>/raw.csv        SMILES + label columns
#   $MOLECULENET_DIR/<task>/manifest.json  source URL, SHA-256, license, fetch date
# and says the fetching script belongs in workflows/. It had never been written, so
# the four sub-tasks the compounds plan asks for (esol, tox21, bace, hiv) plus the
# bbbp default of eval-moleculenet.sh could not be evaluated at all.
#
# Source is DeepChem's S3 mirror, which the cluster reaches directly — the
# netrun.sh DNS workaround the OPV download needed is not required here.
#
# Environment:
#   MOLECULENET_DIR  destination (default: $LEARNING_SOURCE_DIR/eval/moleculenet)
#   SUBTASKS         space-separated (default: esol tox21 bace hiv bbbp)
#   FORCE            set to 1 to re-download tasks that already have raw.csv
#
# Column names are asserted against the specs in data_preparation.py, so a silent
# upstream reshuffle fails here rather than in the middle of an evaluation.
set -euo pipefail

: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR (or set MOLECULENET_DIR)}"
MOLECULENET_DIR="${MOLECULENET_DIR:-${LEARNING_SOURCE_DIR}/eval/moleculenet}"
SUBTASKS="${SUBTASKS:-esol tox21 bace hiv bbbp}"
FORCE="${FORCE:-0}"
BASE="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"

url_for() {
  case "$1" in
    esol)  echo "${BASE}/delaney-processed.csv" ;;
    tox21) echo "${BASE}/tox21.csv.gz" ;;
    bace)  echo "${BASE}/bace.csv" ;;
    hiv)   echo "${BASE}/HIV.csv" ;;
    bbbp)  echo "${BASE}/BBBP.csv" ;;
    *)     return 1 ;;
  esac
}

# One column each side asserts the file is the one the evaluator's spec describes.
smiles_col_for() {
  case "$1" in
    bace) echo "mol" ;;
    *)    echo "smiles" ;;
  esac
}

label_col_for() {
  case "$1" in
    esol)  echo "measured log solubility in mols per litre" ;;
    tox21) echo "NR-AR" ;;
    bace)  echo "Class" ;;
    hiv)   echo "HIV_active" ;;
    bbbp)  echo "p_np" ;;
  esac
}

fetched_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
version="$(date -u +%Y%m%d)"

for task in $SUBTASKS; do
  url="$(url_for "$task")" || { echo "unknown sub-task: $task" >&2; exit 2; }
  dest="${MOLECULENET_DIR}/${task}"
  raw="${dest}/raw.csv"

  if [ -s "$raw" ] && [ "$FORCE" != "1" ]; then
    echo "${task}: raw.csv already present, skipping (FORCE=1 to refetch)"
    continue
  fi

  mkdir -p "$dest"
  tmp="${dest}/.download"
  echo "${task}: fetching ${url}"
  curl -sS -L --fail --max-time 300 -o "$tmp" "$url"

  # The manifest records the checksum of what upstream actually served, so a
  # decompressed raw.csv still points back at a verifiable artefact.
  sha="$(sha256sum "$tmp" | cut -d' ' -f1)"
  bytes="$(stat -c%s "$tmp")"

  case "$url" in
    *.gz) gunzip -c "$tmp" > "$raw" ;;
    *)    mv "$tmp" "$raw" ;;
  esac
  rm -f "$tmp"

  smiles_col="$(smiles_col_for "$task")"
  label_col="$(label_col_for "$task")"
  # HIV.csv ships with CRLF line endings, which would leave a stray \r glued to the
  # last column name and fail the check below for a file that is perfectly fine.
  header="$(head -1 "$raw" | tr -d '\r')"
  for col in "$smiles_col" "$label_col"; do
    case ",${header}," in
      *",${col},"*) ;;
      *) echo "${task}: expected column '${col}' missing from header: ${header}" >&2; exit 3 ;;
    esac
  done
  # Count with a CSV parser, not wc -l: HIV.csv is CRLF with blank lines between
  # records, so line counting reports twice the number of molecules.
  rows="$(python3 - "$raw" <<'PYCOUNT'
import csv, sys
with open(sys.argv[1], newline="") as fh:
    reader = csv.reader(fh)
    next(reader, None)
    print(sum(1 for row in reader if row))
PYCOUNT
)"

  cat > "${dest}/manifest.json" <<JSON
{
  "task": "${task}",
  "name": "MoleculeNet ${task}",
  "home": "https://moleculenet.org/datasets-1",
  "license": "MIT (DeepChem distribution)",
  "version": "${version}",
  "fetched_at": "${fetched_at}",
  "rows": ${rows},
  "smiles_column": "${smiles_col}",
  "notes": "rows counted with a CSV parser; raw.csv is byte-identical to the download except for gzip decompression",
  "files": [
    {"path": "raw.csv", "url": "${url}", "sha256": "${sha}", "downloaded_bytes": ${bytes}}
  ]
}
JSON
  echo "${task}: ${rows} rows -> ${raw}"
done

echo "done: ${MOLECULENET_DIR}"
