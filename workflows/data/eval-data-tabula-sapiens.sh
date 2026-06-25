#!/usr/bin/env bash
# Materialise a Tabula Sapiens cell-type-annotation slice.
#
# CellxGene serves Tabula Sapiens H5ADs via the parent collection
#   https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5
# Individual organ-specific dataset URLs are anonymously reachable
# (HEAD returns 200) — only one specific historic URL was 403'd.
#
# We default to the Tabula Sapiens — Testis slice (~0.39 GB), the
# smallest in the collection. Override TABULA_DATASET_URL to fetch a
# different organ.
#
# Optional environment:
#   TABULA_DATASET_URL   - direct H5AD URL (default: Testis slice)
#   TABULA_TOKENIZER_DIR - HF tokenizer dir (default: rna BERT)
#   TABULA_MAX_CELLS     - cap cells written (default: unset)
#   TABULA_TOP_N_GENES   - top-N expression genes per cell (default: 1024)
#   TABULA_SOURCE        - bypass download; use a pre-downloaded H5AD
#
# Output:
#   $LEARNING_SOURCE_DIR/eval/tabula_sapiens/source.h5ad
#   $LEARNING_SOURCE_DIR/eval/tabula_sapiens/cells.jsonl
#   $LEARNING_SOURCE_DIR/eval/tabula_sapiens/manifest.json
#
# License: CC-BY-4.0 (Tabula Sapiens consortium)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_eval_data_common.sh"

ed_init tabula_sapiens
dest_dir="$(ed_dest)"

URL="${TABULA_DATASET_URL:-https://datasets.cellxgene.cziscience.com/abec77b5-d7b2-4a83-8111-27f4dc8614dd.h5ad}"
H5AD_PATH="${TABULA_SOURCE:-${dest_dir}/source.h5ad}"
TOKENIZER_DIR="${TABULA_TOKENIZER_DIR:-${LEARNING_SOURCE_DIR}/rna/custom_tokenizer_bert}"

cmd=("$PYTHON" -m molcrawl.tasks.evaluation.tabula_sapiens.prepare_jsonl
     --h5ad-url "$URL"
     --h5ad-path "$H5AD_PATH"
     --output-jsonl "${dest_dir}/cells.jsonl"
     --tokenizer-dir "$TOKENIZER_DIR")
if [[ -n "${TABULA_MAX_CELLS:-}" ]]; then
    cmd+=(--max-cells "$TABULA_MAX_CELLS")
fi
if [[ -n "${TABULA_TOP_N_GENES:-}" ]]; then
    cmd+=(--top-n-genes-per-cell "$TABULA_TOP_N_GENES")
fi
"${cmd[@]}"

ed_finalize_manifest \
    "Tabula Sapiens (CellxGene)" \
    "https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5" \
    "CC-BY-4.0" \
    "$(date -u +%Y%m%d)"

cat <<EOF

Next step:
  bash workflows/eval-tabula-sapiens.sh \\
       MODEL_PATH=<rna BERT ckpt> \\
       JSONL_PATH=${dest_dir}/cells.jsonl \\
       MAX_CELLS=200 BOOTSTRAP=20
EOF
