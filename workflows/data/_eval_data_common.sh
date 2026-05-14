#!/usr/bin/env bash
# Common helpers for evaluation dataset download workflows.
#
# Each per-task script under workflows/data/ sources this file and uses
# the helpers below to:
#   * resolve the destination under LEARNING_SOURCE_DIR/eval/<task>/
#   * curl files with optional SHA-256 verification
#   * write a manifest.json capturing source URL, fetch date, version,
#     license, and SHA-256 of every downloaded artefact
#
# Conventions:
#   - Scripts are idempotent: running twice does not re-download files
#     whose checksum matches the manifest entry.
#   - Scripts that need credentials (COSMIC, OMIM, DeepLoc) read them
#     from environment variables and print clear instructions when they
#     are missing instead of failing silently.
#
# Usage from a per-task script:
#
#     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     source "${SCRIPT_DIR}/_eval_data_common.sh"
#     ed_init clinvar
#     ed_download "https://..." "variant_summary.txt.gz" "<sha256>"
#     ed_finalize_manifest "ClinVar" "https://www.ncbi.nlm.nih.gov/clinvar/" "Public Domain"

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate workflows/common_functions.sh so we can reuse check_learning_source_dir
# ---------------------------------------------------------------------------
_ED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_WORKFLOWS_DIR="$(cd "${_ED_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${_WORKFLOWS_DIR}/common_functions.sh"

# Internal state, populated by ed_init / ed_download.
_ED_TASK=""
_ED_DEST=""
_ED_MANIFEST_TMP=""
declare -a _ED_ENTRIES=()

# ---------------------------------------------------------------------------
# Initialise the destination directory and start a fresh manifest buffer.
# Usage: ed_init <task_name>
# ---------------------------------------------------------------------------
ed_init() {
    local task="$1"
    if [ -z "$task" ]; then
        echo "ed_init: task name required" >&2
        exit 1
    fi
    check_learning_source_dir
    _ED_TASK="$task"
    _ED_DEST="${LEARNING_SOURCE_DIR%/}/eval/${task}"
    mkdir -p "${_ED_DEST}"
    _ED_MANIFEST_TMP="$(mktemp)"
    _ED_ENTRIES=()
    echo "[eval-data] task=${_ED_TASK} dest=${_ED_DEST}"
}

# Returns the destination directory for the active task.
ed_dest() {
    if [ -z "${_ED_DEST}" ]; then
        echo "ed_init must be called first" >&2
        return 1
    fi
    printf '%s\n' "${_ED_DEST}"
}

# ---------------------------------------------------------------------------
# Compute SHA-256 of a file.  Falls back across {sha256sum, shasum, openssl}.
# Usage: sha=$(ed_sha256 path)
# ---------------------------------------------------------------------------
ed_sha256() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$path" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$path" | awk '{print $NF}'
    else
        echo "ERROR: no SHA-256 tool available" >&2
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Download a single file into the active task directory with optional
# checksum verification.  When the file already exists and its checksum
# matches the expected value, the download is skipped.
#
# Usage: ed_download <url> <relative_filename> [<expected_sha256>]
# ---------------------------------------------------------------------------
ed_download() {
    local url="$1"
    local rel="$2"
    local expected="${3:-}"
    if [ -z "${_ED_DEST}" ]; then
        echo "ed_init must be called first" >&2
        return 1
    fi
    local out="${_ED_DEST}/${rel}"
    mkdir -p "$(dirname "${out}")"

    if [ -f "${out}" ] && [ -n "${expected}" ]; then
        local current
        current="$(ed_sha256 "${out}")"
        if [ "${current}" = "${expected}" ]; then
            echo "[eval-data] skip ${rel} (sha256 matches)"
            _ed_record_entry "${rel}" "${url}" "${current}"
            return 0
        fi
        echo "[eval-data] checksum mismatch on ${rel}; re-downloading"
    fi

    echo "[eval-data] download ${url} -> ${out}"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --retry-delay 5 \
             --output "${out}.part" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --waitretry=5 -O "${out}.part" "${url}"
    else
        echo "ERROR: neither curl nor wget is available" >&2
        return 1
    fi
    mv "${out}.part" "${out}"

    local sha
    sha="$(ed_sha256 "${out}")"
    if [ -n "${expected}" ] && [ "${sha}" != "${expected}" ]; then
        echo "ERROR: SHA-256 mismatch for ${rel}" >&2
        echo "  expected: ${expected}" >&2
        echo "  actual:   ${sha}" >&2
        return 1
    fi
    _ed_record_entry "${rel}" "${url}" "${sha}"
}

# ---------------------------------------------------------------------------
# Register an already-present file with the manifest (no download).
# Useful when a credentialed download must be performed manually.
#
# Usage: ed_register_existing <relative_filename> <source_url_or_note>
# ---------------------------------------------------------------------------
ed_register_existing() {
    local rel="$1"
    local origin="$2"
    local out="${_ED_DEST}/${rel}"
    if [ ! -f "${out}" ]; then
        echo "ERROR: ed_register_existing: ${out} not found" >&2
        return 1
    fi
    local sha
    sha="$(ed_sha256 "${out}")"
    _ed_record_entry "${rel}" "${origin}" "${sha}"
    echo "[eval-data] registered existing ${rel}"
}

# ---------------------------------------------------------------------------
# Finalise the manifest.json under the task directory.
#
# Usage: ed_finalize_manifest <human_name> <home_url> <license> [<version>]
# ---------------------------------------------------------------------------
ed_finalize_manifest() {
    local name="$1"
    local home="$2"
    local license="$3"
    local version="${4:-unknown}"
    if [ -z "${_ED_DEST}" ]; then
        echo "ed_init must be called first" >&2
        return 1
    fi

    local manifest="${_ED_DEST}/manifest.json"
    local fetched_at
    fetched_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    {
        printf '{\n'
        printf '  "task": "%s",\n' "${_ED_TASK}"
        printf '  "name": "%s",\n' "${name}"
        printf '  "home": "%s",\n' "${home}"
        printf '  "license": "%s",\n' "${license}"
        printf '  "version": "%s",\n' "${version}"
        printf '  "fetched_at": "%s",\n' "${fetched_at}"
        printf '  "files": [\n'
        local n=${#_ED_ENTRIES[@]}
        local idx
        for ((idx = 0; idx < n; idx++)); do
            local entry="${_ED_ENTRIES[$idx]}"
            local sep=","
            if [ "$idx" -eq $((n - 1)) ]; then
                sep=""
            fi
            printf '    %s%s\n' "${entry}" "${sep}"
        done
        printf '  ]\n'
        printf '}\n'
    } > "${manifest}"
    echo "[eval-data] manifest -> ${manifest}"
    rm -f "${_ED_MANIFEST_TMP}"
    _ED_MANIFEST_TMP=""
}

# Internal: append a JSON entry to the in-memory buffer.
_ed_record_entry() {
    local rel="$1"
    local url="$2"
    local sha="$3"
    local entry
    entry=$(printf '{"path": "%s", "url": "%s", "sha256": "%s"}' "${rel}" "${url}" "${sha}")
    _ED_ENTRIES+=("${entry}")
}

# ---------------------------------------------------------------------------
# Print a friendly skip message and exit 0.  Used when a credentialed
# dataset cannot be auto-downloaded.
# Usage: ed_skip_with_instructions <message>
# ---------------------------------------------------------------------------
ed_skip_with_instructions() {
    local message="$1"
    cat <<EOF
[eval-data] SKIP ${_ED_TASK}: ${message}

Refer to docs/04-evaluation/eval_dataset_downloaders.md for credential
setup instructions.  Once the credentials are configured, rerun this
workflow to populate ${_ED_DEST}.
EOF
    exit 0
}
