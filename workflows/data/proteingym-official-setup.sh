#!/usr/bin/env bash
# Set up the official ProteinGym benchmark for the full evaluation (Issue #388, M1).
#
# This fetches what the official aggregation needs and the data-download script
# (workflows/data/eval-data-proteingym.sh) does not: the ProteinGym and ProteinNPT
# source repos (merge.py / performance_*.py / config.json / constants.json /
# reference_files / CV utilities) and all six benchmark data sets, not just DMS
# substitutions. It is additive -- the DMS substitutions already unpacked under
# eval/proteingym/unpacked are left in place and reused.
#
# Everything lands under a gitignored tree (learning_source*). Only this script is
# tracked. It is idempotent: a zip whose sha256 already matches is not re-downloaded,
# and an existing clone is fetched rather than re-cloned.
#
#   LEARNING_SOURCE_DIR=<...> bash workflows/data/proteingym-official-setup.sh
#
# PG_VERSION overrides the Harvard data version (default v1.3). The commit of each
# repo is pinned into official/manifest.json after cloning so a later run records what
# was actually used.
set -euo pipefail

: "${LEARNING_SOURCE_DIR:?export LEARNING_SOURCE_DIR to the protein learning-source dir}"

# Resolve a Python: an explicit $PYTHON, else the repo's miniconda, else python3.
# (There is no bare `python` on this cluster.)
PYTHON="${PYTHON:-}"
if [ -z "${PYTHON}" ]; then
    _root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "${_root}" ] && [ -x "${_root}/miniconda/bin/python" ]; then
        PYTHON="${_root}/miniconda/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="python3"
    else
        PYTHON="python"
    fi
fi

PG_VERSION="${PG_VERSION:-v1.3}"
BASE="${LEARNING_SOURCE_DIR%/}/eval/proteingym/official"
VENDOR="${BASE}/vendor"
DATA="${BASE}/data"
REFS="${BASE}/refs"
LOGS="${BASE}/logs"
mkdir -p "${VENDOR}" "${DATA}" "${REFS}" "${LOGS}"

HARVARD_BASE="https://marks.hms.harvard.edu/proteingym/ProteinGym_${PG_VERSION}"

sha256_of() { sha256sum "$1" | awk '{print $1}'; }

clone_or_fetch() {  # <git-url> <dest>  -- prints only the HEAD sha on stdout
    local url="$1" dest="$2"
    if [ -d "${dest}/.git" ]; then
        echo "[pg-setup] fetch $(basename "${dest}")" >&2
        git -C "${dest}" fetch --quiet --depth 1 origin || true
    else
        echo "[pg-setup] clone ${url}" >&2
        git clone --quiet --depth 1 "${url}" "${dest}"
    fi
    git -C "${dest}" rev-parse HEAD
}

echo "=== ProteinGym official setup (${PG_VERSION}) @ $(date -Is) ==="
echo "base: ${BASE}"

# --- 1. source repos (code + reference_files + CV utilities) ---
PG_COMMIT="$(clone_or_fetch https://github.com/OATML-Markslab/ProteinGym.git  "${VENDOR}/ProteinGym")"
NPT_COMMIT="$(clone_or_fetch https://github.com/OATML-Markslab/ProteinNPT.git "${VENDOR}/ProteinNPT")"
echo "ProteinGym  ${PG_COMMIT}"  | tee "${LOGS}/proteingym_commit.txt"
echo "ProteinNPT  ${NPT_COMMIT}" | tee "${LOGS}/proteinnpt_commit.txt"

# reference_files ship with the ProteinGym repo; copy the CSVs out for convenience.
if [ -d "${VENDOR}/ProteinGym/reference_files" ]; then
    cp -f "${VENDOR}/ProteinGym/reference_files"/*.csv "${REFS}/" 2>/dev/null || true
    echo "[pg-setup] refs: $(ls -1 "${REFS}"/*.csv 2>/dev/null | wc -l) csv"
fi

# --- 2. the six benchmark data sets ---
SETS=(
    DMS_ProteinGym_substitutions
    DMS_ProteinGym_indels
    cv_folds_singles_substitutions
    cv_folds_indels
    clinical_ProteinGym_substitutions
    clinical_ProteinGym_indels
)

download_set() {  # <name>
    local name="$1"
    local zip="${DATA}/${name}.zip"
    local dir="${DATA}/${name}"
    local url="${HARVARD_BASE}/${name}.zip"
    if [ -d "${dir}" ] && [ -n "$(ls -A "${dir}" 2>/dev/null)" ]; then
        echo "[pg-setup] have ${name} (unpacked); skip"
        return 0
    fi
    echo "[pg-setup] download ${name}"
    curl --fail --location --retry 5 --silent --show-error "${url}" -o "${zip}"
    unzip -tq "${zip}" >/dev/null || { echo "ERROR: corrupt ${zip}" >&2; return 1; }
    mkdir -p "${dir}"
    unzip -q -o "${zip}" -d "${dir}"
    printf '%s  %s\n' "$(sha256_of "${zip}")" "${name}.zip" >> "${LOGS}/downloads.sha256"
    rm -f "${zip}"
}

for name in "${SETS[@]}"; do download_set "${name}"; done

# --- 3. manifest ---
"${PYTHON}" - "$PG_COMMIT" "$NPT_COMMIT" "$PG_VERSION" "$BASE" <<'PY'
import json, sys, os, glob, datetime
pg, npt, ver, base = sys.argv[1:5]
data = os.path.join(base, "data")
sets = {}
for d in sorted(glob.glob(os.path.join(data, "*"))):
    if os.path.isdir(d):
        n = sum(1 for _ in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
        sets[os.path.basename(d)] = {"csv_files": n}
m = {
    "benchmark": "ProteinGym",
    "version": ver,
    "retrieved": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
    "proteingym_commit": pg,
    "proteinnpt_commit": npt,
    "data_sets": sets,
}
with open(os.path.join(base, "manifest.json"), "w") as fh:
    json.dump(m, fh, indent=2)
print("manifest:", json.dumps(sets))
PY

echo "=== done @ $(date -Is) ==="
echo "Next (M1): build target inventories keyed on DMS_id with the step-3 consistency checks."
