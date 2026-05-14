#!/usr/bin/env bash
# Download the GUE 28-task benchmark.
#
# Two upstream repos serve this dataset:
#
#   * leannmlindsey/GUE      (default, public, NO token needed)
#       Community mirror of the DNABERT-2 release, kept up-to-date.
#       37 sub-task directories total — a superset of the 28 we score.
#
#   * zhihan1996/DNABERT_2   (canonical, gated; HF_TOKEN required)
#       The original release. Sign in + accept terms at
#       https://huggingface.co/datasets/zhihan1996/DNABERT_2
#       then create a Read-scope token at
#       https://huggingface.co/settings/tokens.
#
# Override behaviour:
#   GUE_HF_REPO      - HuggingFace dataset repo id (default: leannmlindsey/GUE)
#   GUE_HF_REVISION  - branch/tag/commit (default: main)
#   GUE_URL          - explicit zip URL (legacy path; uses curl + auto-unzip)
#
# Output (after run):
#   $LEARNING_SOURCE_DIR/eval/gue/<task>/{train,dev,test}.csv  for each task
#   $LEARNING_SOURCE_DIR/eval/gue/manifest.json
#
# License: see https://huggingface.co/datasets/zhihan1996/DNABERT_2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init gue
dest_dir="$(ed_dest)"

if [ -n "${GUE_URL:-}" ]; then
    # Legacy path: explicit zip URL → curl + unzip.
    out="${dest_dir}/GUE.zip"
    if [ -f "${out}" ] && [ "$(stat -c %s "${out}")" -gt 1000000 ]; then
        echo "[eval-data] GUE.zip already present ($(stat -c %s "${out}") bytes); skipping download"
    else
        echo "[eval-data] download ${GUE_URL} -> ${out}"
        curl_args=(--fail --location --retry 3 --retry-delay 5 --output "${out}.part")
        if [ -n "${HF_TOKEN:-}" ]; then
            curl_args+=(-H "Authorization: Bearer ${HF_TOKEN}")
        fi
        curl "${curl_args[@]}" "${GUE_URL}"
        mv "${out}.part" "${out}"
    fi
    echo "[eval-data] unzipping ${out}"
    unzip_tmp="${dest_dir}/.unzip_tmp"
    rm -rf "${unzip_tmp}"
    mkdir -p "${unzip_tmp}"
    unzip -q -o "${out}" -d "${unzip_tmp}"
    if [ -d "${unzip_tmp}/GUE" ]; then
        src_root="${unzip_tmp}/GUE"
    else
        src_root="${unzip_tmp}"
    fi
    for d in "${src_root}"/*/; do
        [ -d "${d}" ] || continue
        name="$(basename "${d}")"
        [ -d "${dest_dir}/${name}" ] && rm -rf "${dest_dir}/${name}"
        mv "${d}" "${dest_dir}/${name}"
    done
    rm -rf "${unzip_tmp}"
else
    # Default path: snapshot_download from a public or gated HF dataset
    # repo. The python helper handles auth, alias renaming
    # (mirror→canonical), and idempotency.
    GUE_HF_REPO="${GUE_HF_REPO:-leannmlindsey/GUE}"
    GUE_HF_REVISION="${GUE_HF_REVISION:-main}"
    "$PYTHON" -m molcrawl.tasks.evaluation.gue.prepare_csv \
        --hf-repo "${GUE_HF_REPO}" \
        --hf-revision "${GUE_HF_REVISION}" \
        --output-dir "${dest_dir}"
fi

n_tasks="$(find "${dest_dir}" -mindepth 1 -maxdepth 1 -type d -not -name '.*' | wc -l)"
echo "[eval-data] ${n_tasks} sub-task directories under ${dest_dir}"

ed_finalize_manifest \
    "GUE (DNABERT-2 release)" \
    "https://huggingface.co/datasets/zhihan1996/DNABERT_2" \
    "see DNABERT-2 dataset card" \
    "$(date -u +%Y%m%d)"

cat <<EOF

Next step:
  bash workflows/eval-gue.sh \\
       MODEL_PATH=<dnabert2 ckpt> GUE_DIR=${dest_dir} \\
       TASKS="prom_300_all H3" MAX_EXAMPLES=300 BOOTSTRAP=30
EOF
