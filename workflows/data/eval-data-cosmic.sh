#!/usr/bin/env bash
# Download COSMIC mutation tables via the SPA-era download API.
#
# Background — Sanger overhauled the COSMIC download portal somewhere
# around v98.  The legacy ``/cosmic/file_download/<genome>/cosmic/<v>/<file>``
# endpoint that accepted HTTP Basic-Auth now 302-redirects every request
# (authenticated or not) to ``/cosmic/login``.  The new portal at
# ``https://cancer.sanger.ac.uk/cosmic/download/cosmic`` is a Next.js SPA
# backed by NextAuth.js and a presigned-URL service.  This script drives
# that flow directly:
#
#   1. NextAuth credentials login → session cookie
#   2. GET /api/mono/products/v1/downloads/download-file?path=<s3>&bucket=downloads
#      → JSON ``{"url": "<S3 presigned URL valid 1 h>"}``
#   3. GET that URL → tarball containing README + ``*.tsv.gz``
#
# Easiest setup: copy ``.env.example`` to ``.env`` at the repo root,
# fill in the COSMIC_* lines, and re-run this workflow.  ``.env`` is
# gitignored and auto-sourced by ``workflows/common_functions.sh``.
#
# Required environment:
#   COSMIC_EMAIL      - the email registered with COSMIC
#   COSMIC_PASSWORD   - the password registered with COSMIC
#
# Optional:
#   COSMIC_VERSION    - release version, e.g. v100 (default: v100)
#   COSMIC_GENOME     - GRCh38 or GRCh37 (default: GRCh37; the default
#                       Cancer Mutation Census product is GRCh37-only in v100)
#   COSMIC_PROJECT    - one of: cosmic | cell-lines-project |
#                       cancer-mutation-census (default: cancer-mutation-census,
#                       which is the only product carrying FATHMM scores in
#                       v100+ and is what the cosmic evaluator wants)
#   COSMIC_PRODUCTS   - space-separated list of product_codes to fetch
#                       (default: alldata-cmc).  Run with COSMIC_LIST=1 to
#                       see the catalog of authorised products instead of
#                       downloading.
#   COSMIC_LIST       - if set to 1, list catalog and exit without download.
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/cosmic/
#     <product>.tar              # raw tarball
#     <product>/                 # extracted contents (README + .tsv.gz)
#     manifest.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init cosmic

if [ -z "${COSMIC_EMAIL:-}" ] || [ -z "${COSMIC_PASSWORD:-}" ]; then
    ed_skip_with_instructions \
        "set COSMIC_EMAIL and COSMIC_PASSWORD to download COSMIC (free academic registration at https://cancer.sanger.ac.uk/cosmic/register)."
fi

VERSION="${COSMIC_VERSION:-v100}"
GENOME="${COSMIC_GENOME:-GRCh37}"
PROJECT="${COSMIC_PROJECT:-cancer-mutation-census}"
PRODUCTS="${COSMIC_PRODUCTS:-alldata-cmc}"
LIST_ONLY="${COSMIC_LIST:-0}"

COOKIE_JAR="$(mktemp -t cosmic_cookies.XXXXXX)"
trap 'rm -f "${COOKIE_JAR}"' EXIT

# ---------------------------------------------------------------------------
# Step 1: NextAuth credentials login.  Without ``json=true`` the endpoint
# returns a 302 + session-cookie pair; with ``json=true`` you also get a
# JSON body with ``{"url": ...}`` (we don't need it, but it's quieter).
# ---------------------------------------------------------------------------
echo "[eval-data] cosmic: NextAuth login as ${COSMIC_EMAIL}"
csrf=$(curl --fail --silent --show-error \
    -c "${COOKIE_JAR}" \
    "https://cancer.sanger.ac.uk/api/auth/csrf" \
    | python -c 'import json,sys; print(json.load(sys.stdin)["csrfToken"])')

curl --silent --show-error -L -o /dev/null \
    -b "${COOKIE_JAR}" -c "${COOKIE_JAR}" \
    -X POST "https://cancer.sanger.ac.uk/api/auth/callback/credentials?json=true" \
    --data-urlencode "csrfToken=${csrf}" \
    --data-urlencode "email=${COSMIC_EMAIL}" \
    --data-urlencode "password=${COSMIC_PASSWORD}" \
    --data-urlencode "callbackUrl=https://cancer.sanger.ac.uk/cosmic/download/cosmic"

# Verify the session is real.  A failed login returns ``{}`` here.
session_user=$(curl --fail --silent --show-error \
    -b "${COOKIE_JAR}" \
    "https://cancer.sanger.ac.uk/api/auth/session" \
    | python -c 'import json,sys; d=json.load(sys.stdin); print(d.get("user",{}).get("email","") or "")')
if [ -z "${session_user}" ]; then
    echo "[eval-data] COSMIC login failed — verify COSMIC_EMAIL / COSMIC_PASSWORD" >&2
    exit 1
fi
echo "[eval-data] cosmic: logged in as ${session_user}"

# ---------------------------------------------------------------------------
# Step 2: scrape the download portal HTML once to get the catalog.  The Next.js
# server-renders the per-release product list (with s3_object paths and SHA256
# checksums) into the initial RSC stream, so a single GET gives us everything
# we need to address files by ``project / version / product_code``.
# ---------------------------------------------------------------------------
catalog_html="$(mktemp -t cosmic_catalog.XXXXXX.html)"
trap 'rm -f "${COOKIE_JAR}" "${catalog_html}"' EXIT
curl --fail --silent --show-error -L \
    -b "${COOKIE_JAR}" \
    -o "${catalog_html}" \
    "https://cancer.sanger.ac.uk/cosmic/download/cosmic"

resolve_file_py=$(cat <<'PY'
import json, re, sys, html as _html

html_path, project, version, product_code, genome = sys.argv[1:6]
with open(html_path) as f:
    html_text = f.read()
pushes = re.findall(r'self\.__next_f\.push\((\[.+?\])\)</script>', html_text, re.DOTALL)
catalog = None
for raw in pushes:
    try:
        arr = json.loads(raw)
    except Exception:
        continue
    if not (isinstance(arr, list) and len(arr) == 2 and isinstance(arr[1], str)):
        continue
    payload = arr[1]
    m = re.match(r'([0-9a-f]+):', payload)
    if not m:
        continue
    body = payload[m.end():]
    if '"projects"' not in body[:200]:
        continue
    try:
        catalog = json.loads(body)
        break
    except Exception:
        continue
if catalog is None:
    print("ERROR: could not locate COSMIC catalog payload", file=sys.stderr)
    sys.exit(2)
proj = next((p for p in catalog["projects"] if p["project_code"] == project), None)
if proj is None:
    print(f"ERROR: project {project!r} not in catalog (have: "
          f"{[p['project_code'] for p in catalog['projects']]})", file=sys.stderr)
    sys.exit(3)
release_target = int(version.lstrip("v"))
rel = next((r for r in proj["products_per_release"] if r["release_version"] == release_target), None)
if rel is None:
    versions = sorted({r["release_version"] for r in proj["products_per_release"]}, reverse=True)
    print(f"ERROR: version v{release_target} not in {project} (have: {versions})", file=sys.stderr)
    sys.exit(4)
prod = next((p for p in rel["products"] if p["product_code"] == product_code), None)
if prod is None:
    codes = sorted(p["product_code"] for p in rel["products"])
    print(f"ERROR: product {product_code!r} not in {project} v{release_target}\n"
          f"  available: {codes}", file=sys.stderr)
    sys.exit(5)
files = prod.get("download_files") or []
match = next((f for f in files if f.get("genome") == genome), None)
if match is None:
    genomes = sorted({f.get('genome') for f in files})
    print(f"ERROR: genome {genome!r} not in {product_code} v{release_target}\n"
          f"  available: {genomes}", file=sys.stderr)
    sys.exit(6)
sha256 = ""
for c in (match.get("checksums") or []):
    if c.get("algorithm") == "sha256":
        sha256 = c.get("hash", "")
        break
print(json.dumps({
    "filename": match["filename"],
    "s3_object": match["s3_object"],
    "size": match.get("size", "?"),
    "sha256": sha256,
    "is_authorised": match.get("is_authorised", False),
}))
PY
)

list_catalog_py=$(cat <<'PY'
import json, re, sys

html_path = sys.argv[1]
with open(html_path) as f:
    html_text = f.read()
pushes = re.findall(r'self\.__next_f\.push\((\[.+?\])\)</script>', html_text, re.DOTALL)
catalog = None
for raw in pushes:
    try:
        arr = json.loads(raw)
    except Exception:
        continue
    if not (isinstance(arr, list) and len(arr) == 2 and isinstance(arr[1], str)):
        continue
    payload = arr[1]
    m = re.match(r'([0-9a-f]+):', payload)
    if not m:
        continue
    body = payload[m.end():]
    if '"projects"' not in body[:200]:
        continue
    try:
        catalog = json.loads(body)
        break
    except Exception:
        continue
if catalog is None:
    print("could not parse catalog", file=sys.stderr); sys.exit(2)
for proj in catalog["projects"]:
    print(f"\n{proj['project_code']}  ({proj['project_name']})")
    versions = sorted({r['release_version'] for r in proj['products_per_release']}, reverse=True)
    for r in proj["products_per_release"]:
        for p in r["products"]:
            files = p.get("download_files") or []
            if not files:
                continue
            authd = "✓" if any(f.get("is_authorised") for f in files) else " "
            for f in files:
                print(f"  {authd}  v{r['release_version']:>3} {p['product_code']:<35} "
                      f"{f.get('genome',''):<7} {f.get('size','?'):>10}  {f['filename']}")
PY
)

if [ "${LIST_ONLY}" = "1" ]; then
    echo
    python -c "${list_catalog_py}" "${catalog_html}"
    echo
    echo "[eval-data] catalog dump only — exiting before download."
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 3: for each requested product, resolve to (filename, s3_object), ask
# the API for a presigned URL, download into the eval directory, and unpack
# the tarball so downstream loaders can read the .tsv.gz directly.
# ---------------------------------------------------------------------------
for product_code in ${PRODUCTS}; do
    echo
    echo "[eval-data] cosmic: resolving ${PROJECT}/${VERSION}/${product_code} (${GENOME})"
    file_meta=$(python -c "${resolve_file_py}" \
        "${catalog_html}" "${PROJECT}" "${VERSION}" "${product_code}" "${GENOME}")
    filename=$(printf '%s' "${file_meta}" | python -c 'import json,sys; print(json.load(sys.stdin)["filename"])')
    s3_object=$(printf '%s' "${file_meta}" | python -c 'import json,sys; print(json.load(sys.stdin)["s3_object"])')
    expected_sha=$(printf '%s' "${file_meta}" | python -c 'import json,sys; print(json.load(sys.stdin)["sha256"])')
    size_str=$(printf '%s' "${file_meta}" | python -c 'import json,sys; print(json.load(sys.stdin)["size"])')
    is_auth=$(printf '%s' "${file_meta}" | python -c 'import json,sys; print(json.load(sys.stdin)["is_authorised"])')
    echo "[eval-data] cosmic:   file=${filename}"
    echo "[eval-data] cosmic:   size=${size_str} sha256=${expected_sha:0:12}... auth=${is_auth}"
    if [ "${is_auth}" != "True" ]; then
        echo "[eval-data] cosmic: NOT AUTHORISED for ${product_code}." >&2
        echo "[eval-data]   Your COSMIC license tier may not include this product." >&2
        echo "[eval-data]   Skip flag check or upgrade at https://cancer.sanger.ac.uk/cosmic/license." >&2
        continue
    fi

    out_tar="${_ED_DEST}/${filename}"
    if [ -f "${out_tar}" ] && [ -n "${expected_sha}" ]; then
        current_sha=$(ed_sha256 "${out_tar}")
        if [ "${current_sha}" = "${expected_sha}" ]; then
            echo "[eval-data] cosmic:   already downloaded (sha256 matches)"
            ed_register_existing "${filename}" "https://cancer.sanger.ac.uk/cosmic/download/cosmic#${PROJECT}-v${VERSION#v}-${product_code}"
            continue
        fi
    fi

    echo "[eval-data] cosmic:   requesting presigned URL"
    presigned=$(curl --fail --silent --show-error \
        -b "${COOKIE_JAR}" \
        --get \
        --data-urlencode "path=${s3_object}" \
        --data-urlencode "bucket=downloads" \
        "https://cancer.sanger.ac.uk/api/mono/products/v1/downloads/download-file" \
        | python -c 'import json,sys; print(json.load(sys.stdin)["url"])')

    ed_download "${presigned}" "${filename}" "${expected_sha}"

    # Unpack the tar into a sibling directory for downstream loaders.
    unpack_dir="${_ED_DEST}/${product_code}"
    mkdir -p "${unpack_dir}"
    if ! tar -xf "${out_tar}" -C "${unpack_dir}"; then
        echo "[eval-data] cosmic:   WARNING: tar -xf failed for ${filename}" >&2
    fi
    echo "[eval-data] cosmic:   unpacked to ${unpack_dir}"
    ls -1 "${unpack_dir}" | sed 's/^/[eval-data] cosmic:     /'
done

ed_finalize_manifest \
    "COSMIC (${PROJECT} ${VERSION})" \
    "https://cancer.sanger.ac.uk/cosmic/download/cosmic" \
    "Academic / Commercial dual — see https://cancer.sanger.ac.uk/cosmic/license" \
    "${VERSION}"
